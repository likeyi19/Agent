# Cell-type annotation

Researchers can upload scATAC H5AD or select accepted count results and request native cell-type annotation through the ordinary conversation. MAESTRO Enhanced RP is one supported method; native reference-to-query label transfer remains available separately. The Registry and scientific owners select appropriate supported operations. Native scores are computational evidence, not calibrated biological confidence.

## Inputs and native method

`annotate_scATAC_cell_types` reuses pinned MAESTRO Enhanced regulatory potential
(RP), native group-marker calculation and signature scoring. Its direct region
method `maestro-enhanced-region-counts.v2` accepts explicitly identified raw peak
counts, fragment-overlap counts and binary accessibility for human/hg38 or
mouse/mm10 with compatible transcript/exon and signature resources. The original
human Cell Ranger cut-site method and accepted canonical-matrix method retain
their identities and historical readers. This extension does not reinterpret old
publications or binarize supplied fragment counts.

The accepted PBMC10k Gate 1 run used MAESTRO Enhanced RP on the original,
non-binarized `fragment_counts` in `X` under `maestro-enhanced-region-counts.v2`,
followed by native group-marker calculation and signed marker-signature scoring.
This is the recorded count-weighted execution profile. The
[original MAESTRO paper](https://link.springer.com/article/10.1186/s13059-020-02116-x)
describes its RP formulation with a binary peak-occurrence matrix; the accepted
count-weighted run must not be described as reproducing that binary-input method.
Its scores and marker evidence remain computational, dependent annotation evidence.
MAESTRO is a supported method whose suitability depends on the inputs and references,
not an exclusive or universally preferred annotation method.

The [MAESTRO implementation](https://raw.githubusercontent.com/liulab-dfci/MAESTRO/74f10babf4125b98ef005df4186b14ab02e13bfd/MAESTRO/scATAC_Genescore.py)
weights region measurements by transcript/exon and distance relationships.
[MAESTRO documentation](https://liulab-dfci.github.io/MAESTRO/) describes raw and
binary peak input. Numerical admissibility alone does not establish the best
biological representation, reference relevance or cell identity.

Required scientific information is the measurement history, species/assembly,
exact zero-based `chrom:start-end` feature identities, compatible references, and
independent cell groups. Groups may come from an explicitly selected `obs` column,
exact ordered TSV, or an accepted dedicated or native Leiden result whose recorded ancestry
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

Storage support includes canonical CSR/CSC and ordinary dense arrays converted
to sparse. Complete sparse matrices are never densified. Nonempty unique axes,
exact source identity, supported HDF5 encoding, exact integral counts and native
method applicability remain required. Native zero-weight and unmatched-contig
participation is reported. Agent no longer imposes cell/feature, NNZ, encoded-file,
decoded-matrix or dense-conversion size caps on this scientific invocation.

Python preparation, native MAESTRO RP, its R child processes and publication
validation run inside the [shared native execution boundary](native-execution.md).
The operator must configure `AGENT_NATIVE_MEMORY_BYTES`; there is no universal
memory default or dataset-based admission estimate. Kernel cgroup v2 limits cover
the aggregate task, with swap disabled. Missing containment fails operationally
before expensive scientific reads. The obsolete `AGENT_ANNOTATION_SOURCE_BYTES`,
`AGENT_ANNOTATION_LOGICAL_BYTES` and `AGENT_ANNOTATION_NONZEROS` policy is removed.
The existing optional R timeout remains finite, nested within the whole-task
operator timeout. Browser upload and scientific delivery budgets remain separate.

## Selected-method input preparation

The normal Planner selects annotation and its scientific source before the
application resolves method-specific inputs. A scoped compiler callback in
`annotation_preparation` binds the existing ScientificResourceLibrary base
reference using `annotation_input.resource_specification`. It specializes the
actual declared measurement and biological context; the template's historical
Cell Ranger measurement does not override current fragment counts. Explicit
source, count slot, method, grouping and reference choices retain precedence.
Missing biology or ambiguous references require clarification. Internal spec paths
and checksums are application products, not researcher prerequisites.

An explicit accepted grouping is a scientific input selection, independently of
the Interpreter's coarse `context` field. Its exact accepted ancestry is lowered
to the shared captured-result input selection. The existing planning projection
exposes qualified prior-output ports only for those selected results; unrelated
current outputs and figure references grant no additional scientific inputs.
Preparation and execution use the same selection owner and captured Revision.
An already planned grouping binding is accepted when its authenticated grouping
identity agrees with the explicit selection; a conflicting grouping still fails.

Selected-method admission v3 retains attachment candidates separately from the
uploaded datasets this task consumes. An accepted-result-only task has no upload
consumption guard. Selecting a previous upload establishes its new consumer role
without another upload; a pending task retains its exact selected sources/roles.
Uploaded reference-only files retain their exact registration and byte checks,
and the existing scientific resource owner checks their pinned method inputs.
Unselected historical files impose no current consumption requirement. Older
admissions remain readable with their original attribution and saved responses.

Uploads and accepted sources use this same owner. Private task specifications are
immutable adapter bytes, never registered as fake uploads. The existing scientific
grouping owner consumes exact accepted analysis files and any required ordered TSV
adaptation; native groups retain their accepted dependency proofs.
Accepted group association requires exact count ancestry and ordered cells. Native
function identity, actual written slots and accepted dependencies distinguish newly
computed groups from inherited fields and independent returns. The old attachment-only
annotation preparation writer is removed; narrowly scoped replay readers preserve
saved historical declarations and accepted results.

One invocation-local prepared input serves native execution and immediate
publication checks. Independent derivative checks compare exact arrays in bounded
chunks and preserve axes, metadata, layers and ancillary scientific slots. Recovery
may qualify the source again, but never reruns native RP, markers or scoring.

## Completing grouping prerequisites

A researcher may request cell-type annotation without naming tools,
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

## Native activity figures and reporting

SRV-1B adds automatic composition, positive-native-marker RP dot plots and genuine
per-cell RP distributions through the existing verified visualization owner.
Accessibility-derived regulatory potential is not measured RNA expression, and
markers used in annotation are dependent evidence. Accepted compatible UMAP
coordinates can be recolored without refitting. Omitted groups/genes are explicit;
no additional annotation or marker test runs in the renderer.

The accepted factual report displays data context, method, group/label counts and assigned/unresolved populations independently of optional generated prose. Native RP, marker and signature details, effective parameters and source identities remain available in evidence and methods; source matrix and analysis sections own their feature counts and preprocessing/clustering facts. Download the annotated H5AD and useful scientific figures.

Agent ends this workflow at its own native annotations and supporting evidence. Label renaming, ontology alignment and predicted-versus-reference label benchmarks are external research tasks. They are not Agent execution modes or resource requirements. The independent Gate 2 evidence remains outside the product.
