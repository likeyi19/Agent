# UA3.5.4d — Practical scientific capability readiness and priority selection

UA3.5.4d completes a read-only readiness assessment of the **23 registered tools
and 172 public parameters**. It reuses the retained parameter survey and existing
acceptance; it does not qualify all tools with a live model. The preferred next
task is **connect the original human EpiZoo resources to the existing Agent
catalog/default selector**. This is a configuration integration for the already
verified joint human–mouse checkpoint, not qualification of another model.
No selected repair is implemented here. UA3.5 remains open; UA3.6 has not begun.

## Baseline, scope and preservation

On 2026-10-10, entry and final branch are `main`. HEAD, local `main` and local
`origin/main` equal `8bcad5b42cb812c89dfc232e3ceadea2fc5e44a7`.
The index is clean and byte-identical. Entry has **30 modified tracked files,
24 new non-evaluation files**, and the existing untracked evaluations. Final
state adds only this report: 30 modified tracked files and 25 new such files.
All UA3.5.1–UA3.5.4c work remains unstaged and unchanged.

The entry inventory is `/tmp/agent-ua354d-preservation-6h6ec749`.
All **588 original repository files**, **1,002 existing evaluation files**,
198 inventoried scientific outputs, 208 inventoried real-state files and the
operator configuration retain SHA-256, size, mtime and mode. The original
5,231,645,507-byte checkpoint retains its recorded size/mtime/mode; its established
digest is reused without rehashing or loading that binary. No reset, clean, stage,
fetch, commit or push occurs. No EpiZoo source, dataset, scientific default,
authority contract, operator configuration or production code changes.

Inspection is limited to Registry metadata, the retained survey, accepted reports,
selected input/resource/authority boundaries and the concrete human-resource
question. Small human auxiliary files and retained small evidence are checked
against existing hashes. No scientific owner is executed; no test suite, live
provider/catalog lookup, GPU/model load, browser or scientific reconstruction runs.
Disposable inspection records stay outside Git. The reports reused include
[UA3.5.3](ua3.5.3-unified-scientific-input-binding.md),
[UA3.5.4a](ua3.5.4a-live-semantic-qualification.md),
[UA3.5.4b](ua3.5.4b-prerequisite-handoff.md) and
[UA3.5.4c](ua3.5.4c-output-selection-contract.md).

## Inventory reconciliation and common mechanism

The original survey was returned in chat, as its request required, rather than
saved as a repository report. Its exact final response is retained at
`/home/likeyi/.codex/sessions/2026/10/09/rollout-2026-10-09T15-34-01-01a11f95-3b89-7c12-95eb-47afc2e085fd.jsonl`,
line 1518, timestamp `2026-10-10T00:46:23.842Z`. The
[extracted response](/tmp/agent-ua354d-retained-survey-yz1cixos/survey-response-1.txt)
has SHA-256 `3dd8a8dddb2d3a8451366972a7553317959bf2582d8a93cbad7dfd60628e1bdb`;
[provenance](/tmp/agent-ua354d-retained-survey-yz1cixos/provenance.json)
identifies its source. No replacement 172-row survey is generated.

An exact name/requiredness comparison with the current
[Registry](../src/agent/orchestration/registry.py) finds **zero new or removed
tools/parameters and zero requiredness changes**: 23 tools, 124 required and
48 optional arguments. Every argument has planning metadata and a semantic
consumer-field mapping. The retained handling partition remains:

| Primary handling category in the survey | Required | Optional | Total |
| --- | ---: | ---: | ---: |
| Internal deterministic binding | 83 | 8 | **91** |
| User or authoritative scientific declaration | 27 | 9 | **36** |
| Reuse existing owner default | 0 | 30 | **30** |
| Resolve qualified resource | 12 | 1 | **13** |
| Empirical candidate requiring qualification | 2 | 0 | **2** |
| Total | **124** | **48** | **172** |

These are the survey's primary handling categories, not Registry enums or counts
of working Web integrations. All 48 optional arguments have omission metadata;
some are resource choices, conditional declarations or internal bindings rather
than the survey's 30 ordinary defaults. Likewise, the 58 `scientific_parameter`
flags include resources and declarations, not blanket clarification eligibility.
[Reconciliation evidence](/tmp/agent-ua354d-survey-reconciliation-sevnc9x7/reconciliation.json)
and [current metadata](/tmp/agent-ua354d-registry-readiness-c2ap8jen/registry-readiness.json)
record these distinctions.

The shared numerical declaration projection currently offers **14 parameters
across five tools**:

| Tool | Offered parameters |
| --- | --- |
| `build_cell_neighbors` | `n_neighbors`, `random_seed` |
| `cluster_cells` | `resolution`, `random_seed` |
| `compute_cell_umap` | `min_dist`, `spread`, `random_seed` |
| `transfer_cell_labels` | `n_neighbors`, `min_confidence` |
| `select_scATAC_cells` | `min_qc_fragment_records`, `min_tss_enrichment`, `min_tss_flank_evidence`, `max_qc_fragment_records`, `max_nucleosome_signal` |

Only the first two Selection thresholds are required. The species prerequisite
path separately covers `epizoo_embed_cells` on `h5ad-science.v1`; it does not
automatically collect every tool's species/assembly declaration. The other
declarations remain usable through complete typed application inputs or operator
InputSets, but are not all recoverable from prose through the current binder.

One Interpreter and scoped Planner propose intent, tools, sources and outputs.
Registry/ToolSpec, typed binding, compiler, preflight, Session/Revision and the
scientific owners admit or reject those proposals. A registered tool needs no
separate language parser. Registration alone supplies neither missing scientific
facts nor evidence that the live model consistently selects a valid operation.

## Existing capability families and workflow relationships

The current [capability index](../src/agent/orchestration/planning_scope.py) has
eight families. There are 24 memberships because `epizoo_embed_cells` belongs
to both embedding analysis and reference annotation. Family codes below are
only abbreviations for this report; the canonical IDs remain unchanged.

| Code | Canonical capability ID | Tools |
| --- | --- | ---: |
| PI | `processed_inspection` | 1 |
| EA | `embedding_analysis` | 5 |
| RA | `reference_annotation` | 3 |
| DA | `differential_accessibility` | 3 |
| RP | `raw_preprocessing` | 7 |
| MA | `exact_matrix_adoption` | 1 |
| AN | `marker_annotation` | 1 |
| SA | `species_adaptation` | 3 |

Existing relationships identify the following shared prerequisites:

- Registered FASTQ/BAM can enter `inspect_raw_scATAC`; qualified intake,
  source-bound library context and reference feed their respective fragment
  producers. Explicit external fragments enter `import_scATAC_fragments`.
  Producer-neutral verified v2 fragments then feed QC, explicit Selection and
  canonical cell-by-cCRE construction. These are legal choices, not an automatic
  recipe or instruction to manufacture missing stages.
- A compatible H5AD can be inspected and embedded with EpiZoo; its ordered
  embedding/ID pair feeds neighbors, then Leiden and UMAP. Fixed clustering
  evaluation also needs independent ordered ground truth and its declared column.
- Reference annotation uses exact reference/query H5ADs and their embedding
  bundles with matching species/checkpoint provenance. Transfer does not require
  clustering or UMAP; annotation evaluation needs separate ground truth.
- Feature-space validation feeds exact replicate pseudobulk, then declared
  replicate-aware DA. Matrix semantics, replicate identities and contrast/design
  are scientific declarations, not defaults inferred from available columns.
- Exact canonical adoption is distinct from canonical fragment-derived history.
  MAESTRO annotation consumes the accepted canonical fragment-count route and
  its explicit groups/context/resources. Neutral external adoption or neutral
  fragment-derived matrices use their own reference contracts; compatible neutral
  matrices may feed explicitly specified species adaptation. M14.10's qualified
  neutral QC-selected matrix route supersedes M14.7's earlier explicit-cells-only
  limitation. Neutral BAM preparation remains a data-layer API, not another
  registered public tool. Adaptation publishes a model; general adapted-target
  inference is not a registered continuation supplied by this assessment.

## 23-tool readiness matrix

**S** means registered and structurally representable using the existing semantic
ports and complete typed prerequisites. It is not execution readiness for an
arbitrary file. **D** adds demonstrated scripted conversational binding/execution
in the cited Application/Web scope. **L-plan** is older live Planner-only
PLAN_ONLY evidence; **L-admit** is live interaction/planning/admission;
**L-science** includes live semantics and verified scientific execution.
Historical scientific execution and scripted browser acceptance are identified
separately. For rows without an L label, that live conversational path is **not
yet demonstrated** by the reviewed evidence; that absence alone is not a defect.

| Canonical tool / family | Conversational contract | Input and prerequisite readiness | Defaults and resources | Existing execution evidence | Concrete limitation / responsible owner |
| --- | --- | --- | --- | --- | --- |
| `inspect_scATAC` / PI | S, D; L-plan | Registered/uploaded or explicit H5AD; inspection grants no matrix/model authority. | No scientific default or model needed. | Real mouse chain in [UA3.2.2](ua3.2.2-h5ad-composition.md); scripted Web/browser; live Groq inspection plan/preflight in [scoped acceptance](../AGENTS.md#capability-scoped-two-stage-planning). | No numerical blocker. Structure observation does not establish biological declarations or model readiness: inspection owner. |
| `epizoo_embed_cells` / EA, RA | S, D species; L-admit mouse | Sparse count-like full human/mouse axis, exact retained feature names, explicit species and registered identity. | Same verified joint checkpoint; original auxiliaries for both species exist. Mouse mapped default; human catalog/default connection absent. | Historical real mouse [UA3.2.2](ua3.2.2-h5ad-composition.md); real human canonical C0 [M12](m12-matrix-level-input-closeout.md). UA4c live mouse admission with all tools withheld. | **Human mapping gap: operator catalog/Application.** Current pinned human Web execution not yet demonstrated. Generic prior-output reuse/export of embeddings lacks its required owner contract. |
| `build_cell_neighbors` / EA | S, D chain; numeric arguments structurally offered; L-admit in UA4c plan | Exact 512-dimensional embeddings and ordered ID sidecar, from same-plan output or complete direct inputs. | Owner k=15, Euclidean, seed=0; k requires at least 16 cells. | Real 2,000-cell M6.1 and 32-cell [UA3.2.2](ua3.2.2-h5ad-composition.md); scripted HTTP science. | No new default needed. Applicable cell count and artifact integrity remain owner checks; generic cross-turn authority absent. Explicit numeric live binding not yet demonstrated. |
| `cluster_cells` / EA | S, D; **L-science** explicit/default/paraphrase | Genuine neighbors H5AD with exact cell/order/provenance. | Owner resolution=1.0, seed=0; explicit scoped values supported. | [UA4b B1–B3](ua3.5.4b-prerequisite-handoff.md): real live tiny Leiden plus normal verification; prior real mouse chains. | Generic cross-turn graph authority/export absent. Live invalid/repeated-consumer probes stop at Interpreter and do not qualify those downstream rejection paths. |
| `compute_cell_umap` / EA | S, D chain/numeric seam; L-admit in UA4c plan | Exact supported clustered compact H5AD; no raw-feature densification. | Owner 2D, min_dist=0.5, spread=1.0, seed=0; coupled validity stays with owner. | Real M6.1/[UA3.2.2](ua3.2.2-h5ad-composition.md); scripted Web chain. | Generic cross-turn compact-analysis authority/export absent. Live numerical override/execution not yet demonstrated. |
| `evaluate_cell_clustering` / EA | S; required text key through structured inputs | Fixed clustering/UMAP plus separate ground-truth H5AD with identical ordered cells and explicit `label_key`. | Owner `cluster_key=leiden`; no inferred label column or metric tuning. | Real fixed 2,000-cell M6.2 evaluation and independent recomputation in [AGENTS](../AGENTS.md#milestone-62--quantitative-clustering-evaluation). | Label key is excluded from current conversational binder; evaluator upload composition is absent. Application binding/composition owners; science already checks column/order. |
| `transfer_cell_labels` / RA | S; numeric arguments structurally offered; complete transfer conversation not yet demonstrated | Exact reference/query H5ADs, embeddings/IDs, declared reference label, same species and historical checkpoint. | Owner k=20, Euclidean, min_confidence=0; requires at least 20 reference cells. | Real held-out 1,400→600 M6.3 transfer, accepted deterministic planning/verification in [AGENTS](../AGENTS.md#milestone-63--reference-to-query-cell-label-transfer). | Required `reference_label_key` lacks prose binding; ordinary selected-upload composition is not general multi-H5AD composition. No automatic cross-species transfer or calibrated confidence: owner. |
| `evaluate_cell_annotation` / RA | S; required text key through structured inputs | Fixed transferred annotation and independent exact-order query ground truth; explicit `ground_truth_label_key`. | Fixed evaluation conventions; no scientific key default. | Real 600-cell fixed M6.4 evaluation in [AGENTS](../AGENTS.md#milestone-64--annotation-evaluation-and-confidence-diagnostics). | Required key lacks prose binding; current upload companion does not compose this evaluator. Application. Generic historical annotation-output authority/export absent. |
| `validate_scATAC_feature_space` / DA | S; declarations through structured inputs | Sparse source with explicit matrix source/value semantics, species/assembly and optional coordinate declarations. | Conditional layer/coordinate keys; omission is not evidence that X is raw counts. | M8.1 realistic backed-sparse synthetic production/independent verification in [AGENTS](../AGENTS.md#milestone-81--feature-space-and-replicate-aware-pseudobulk-foundation). | Broad declarations/upload composition are outside current scalar handoff. Owner rejects normalized-continuous values despite their descriptive Registry choice; no normalization repair. |
| `build_replicate_pseudobulk` / DA | S; metadata roles through structured inputs | Verified feature space; genuine replicate/group/condition columns; declared group source; exact compatible annotation if requested. | Empty covariates is an existing omission choice, not proof of no confounding. | M8.1 1,024×50,000 sparse acceptance, 64 exact SUM pseudobulks in [AGENTS](../AGENTS.md#milestone-81--feature-space-and-replicate-aware-pseudobulk-foundation). | Roles cannot be guessed; string/list declarations lack general conversational collection. Suitable local real experiment not demonstrated; data/experiment owner. |
| `run_replicate_differential_accessibility` / DA | S; contrast/design through structured inputs | Verified count pseudobulk and valid biological replication, directed contrast, design and covariates. | Existing pinned edgeR v4 QL/R implementation; no LLM statistical settings. | Accepted M8.2 software/synthetic fitting, independent verification and Application reporting in [AGENTS](../AGENTS.md#milestone-82--replicate-aware-differential-accessibility). | No eligible local real two-condition raw-count experiment in accepted audit. Direct `condition_key` repetition remains a reviewed metadata-handoff opportunity. No fabricated replication; scientific owner. |
| `inspect_raw_scATAC` / RP | S, D uploaded intake | Registered BAM or explicitly complete FASTQ collection; declaration omissions can truthfully produce non-readiness. | No producer/model resource needed for inspection; optional declarations retain unresolved semantics. | Real PBMC FASTQ transport/bounded intake and scripted native browser in [UA3.4.3](ua3.4.3-fastq-web-upload.md); synthetic BAM Web acceptance. | Successful intake is not producer readiness. Assembly, assay and barcode-history authority remain explicit; no raw→EpiZoo direct edge. |
| `prepare_scATAC_fragments` / RP | S, D uploaded tiny chain | Exact intake, source-bound FASTQ library/barcode context, approved reference, qualified Chromap/index/runtime. | Fixed accepted producer settings; no public scientific tuning defaults missing. | Real human PBMC FASTQ production [M11.8](m11.8-biological-closeout.md); scripted uploaded tiny owner chain/browser [UA3.4.3](ua3.4.3-fastq-web-upload.md). | Current operator lacks FASTQ companion connection; source-specific context cannot be copied from old locations. Operator/library owner. Real uploaded-source/full mouse biology not supplied by those checks. |
| `import_scATAC_fragments` / RP | S, D | Registered source, declared profile/namespace, approved human/mouse reference; exact optional BGZF/TBI pair. | `source_selection=unknown` truthfully preserves absent history; index is optional. | Genuine tiny import/conservation and scripted HTTP/browser [UA3.4.1](ua3.4.1-external-fragments-web-upload.md). | Current operator lacks fragments companion connection. Arbitrary-species public import and real external biological generalization are not established by this route. |
| `prepare_scATAC_bam_fragments` / RP | S, D | Exact source/intake, qualified paired-ATAC corrected-CB context, source profile and compatible reference. | Fixed accepted filtering/shift policy; no inferred corrected-CB authority. | Genuine tiny preparation/verification and scripted HTTP/browser [UA3.4.2](ua3.4.2-bam-web-upload.md). | Current operator lacks BAM companion connection; source/history eligibility stays with owners. Real BAM biology remains unqualified. UA4c's uncaptured fixture exception remains unresolved. |
| `compute_scATAC_qc` / RP | S, D raw chains | Verified v2 fragments plus matching qualified parent/TSS/QC reference and production catalog/runtime. | Qualified human GENCODE49/mouse M25 resources; separately qualified macaque resource. Fixed metric definitions. | Real human [M11.8](m11.8-biological-closeout.md) and macaque [M14.11](m14.11-real-macaque-qc.md); scripted Web raw chains. | No general resource discovery; exact parent/catalog applicability required. QC computes metrics, neither selects nor statistically calls cells. Operator/QC owner. |
| `select_scATAC_cells` / RP | S, D; **L-science** missing/partial/correction chain | Exact verified QC manifest/digest and explicit depth/TSS minima. Existing accepted authority supports continuation. | Two required minima have no default; three optional conditions omitted/null disable. | [UA4b C1–C4](ua3.5.4b-prerequisite-handoff.md) live tiny owner execution/independent verification; real scripted PBMC Session/browser [M15.6](m15.6-real-multiturn-closeout.md), [M18.5](m18.5-real-interactive-acceptance.md); real macaque science. | **No initial-threshold handoff defect remains.** 1000/4/20 policy unqualified; statistical calling/purity not assessed. Scientific policy owner. |
| `build_scATAC_cell_by_ccre` / RP | S, D exact continuation | Verified fragments/selection and matching complete canonical reference; exact accepted authority reusable. | Fixed int64 canonical record counting; no inferred selection or downstream completion. | Real human [M11.8](m11.8-biological-closeout.md), scripted real Session continuations [M15.6](m15.6-real-multiturn-closeout.md), accepted matrix delivery. | Raw context connection affects new uploaded tasks. Arbitrary peaks/assemblies cannot be repaired into canonical vocabulary; matrix/reference owners. Live matrix conversation not yet demonstrated. |
| `adopt_scATAC_cell_by_ccre` / MA | S; complete typed adoption declarations | Strict axis-only int64 CSR H5AD, full exact human/mouse reference, source/reference hashes and declared semantics. | No adoption defaults; registration alone is not adoption authority. | Tiny Application adoption→inspection, complete conservation/lifecycle [M12.1](m12.1-exact-external-cell-by-ccre-adoption.md); accepted matrix delivery. | Ordinary arbitrary annotated/float/peak H5AD is outside the strict contract. No fabricated raw/QC history or automatic model readiness: adoption owner. |
| `annotate_scATAC_cell_types` / AN | S; complete pinned specification | Accepted canonical fragment-count matrix RunStore authority, exact independent groups, context and resources. | Existing qualified MAESTRO/R owner; no universal tissue/signature default. | Real 1,119-cell PBMC scripted semantic-v4 Application, production plus owner replay [M13.5](m13.5-agent-annotation-integration.md). | Exact context/groups required; external-adopted/neutral/insertion matrices unsupported here. Other tissue/species need applicable qualification, not inferred labels. |
| `adopt_scATAC_cell_by_features` / SA | S; complete typed bindings | Exact constrained external matrix, neutral reference and explicit value semantics; conservation produces its own authority. | Explicit species/taxonomy/assembly/reference; no arbitrary vocabulary/default guess. | Real scripted Application macaque and zebrafish adoption [M14.4](m14.4-real-macaque-mapped-acceptance.md), [M14.5](m14.5-real-zebrafish-de-novo-acceptance.md); accepted matrix delivery. | Neutral adoption does not authorize legacy EpiZoo/MAESTRO or raw/QC history. General uploaded-input composition is not established: Application/owner. |
| `adapt_epizoo_species` / SA | S; explicit strategy/specification | Compatible accepted neutral matrix, qualified original source/SEAM bundles, target reference and explicit mapping/profile where applicable. | Existing closed qualification profiles; production-candidate schedule remains unqualified. | Real bounded mapped macaque and binary de-novo zebrafish scripted Application, strict reload/finite forward and recovery [M14.4](m14.4-real-macaque-mapped-acceptance.md), [M14.5](m14.5-real-zebrafish-de-novo-acceptance.md). | Production convergence/biology and general public adapted-target inference absent. These require distinct owner-approved scope; no strategy/resource inference. |
| `build_scATAC_cell_by_features` / SA | S; exact resource bindings | Verified neutral fragments/reference plus explicit ordered cells, or current qualified neutral QC-selected route; exact primary scope where required. | Fixed shared counting; explicit cells are an existing resource API, not a new tool. | Real 331-cell [M14.8](m14.8-real-macaque-fragment-matrix.md), real 3,220×615,873 QC-selected [M14.11](m14.11-real-macaque-qc.md); scripted public planning and matrix delivery. | Neutral resource provisioning remains explicit; no arbitrary-species registered raw producer or automatic cell calling. Matrix validity alone does not prove adaptation eligibility. |

The matrix does not combine independent acceptance records into an unobserved
live-provider + browser + science pass. In particular, UA4c selects outputs
across EpiZoo/neighbors/Leiden/UMAP, but withholds the whole plan before all tools.
Its two admission passes are not four scientific executions. Its persisted
`Run.preflight_verification` remains null; the executor's separate observation
and Planner trace establish only their respective preflight checks.

## Cross-cutting input and authority findings

Complete noncompanion
[operator InputSets](../src/agent/web/config.py) and direct application inputs
can already supply the structured declarations/resources for registered tools.
Selected uploads instead use reviewed composition families in
[local_resources.py](../src/agent/application/local_resources.py).
The H5AD companion covers EpiZoo/neighbors/Leiden/UMAP, not evaluator,
pseudobulk or general multi-H5AD composition. A string binder extension alone
would not extend those upload routes. FASTQ/BAM/fragments companions deliberately
cannot replace captured source identities or invent library history.

Current read-only operator inspection finds two InputSets: one ordinary PBMC
input and one mouse H5AD species companion. There are **zero activated fragments,
BAM or FASTQ companions** and exactly one EpiZoo resource/default, for mouse.
The default planning profile remains the operator's Groq profile.
[Safe configuration inventory](/tmp/agent-ua354d-preservation-6h6ec749/operator-readiness.json)
records only IDs, flags and input-key names. Thus resources that exist in the
scientific layer are not automatically connected to every new Web upload.

Intra-plan `StepOutputRef` and generic cross-turn `PriorOutputRef` are different
contracts. [bind_output](../src/agent/orchestration/prior_outputs.py) requires a
successful exact RunStore anchor and reusable schema-2 authority.
The existing [scientific authority adapters](../src/agent/tools/data/scientific_authority.py)
cover fragment/QC/selection/matrix/MAESTRO/adaptation publications. They do not
cover ordinary EpiZoo embedding, neighbors, Leiden, UMAP, evaluation, label
transfer, feature-space/pseudobulk or DA outputs. Their legal within-plan edges
and accepted owner executions therefore do not establish generic cross-turn
reuse. This is a concrete authority integration boundary, not missing LLM
training or permission to weaken validation.

Likewise, [accepted result delivery](ua3.3.2-accepted-matrix-delivery.md) covers
the four matrix tools and QC/Selection tables, not every registered result.
Embedding/ID pairs, compact analysis H5ADs, evaluation JSON and other deferred
products need their reviewed payload identity, authority/publication and privacy
contracts before a transport allowlist can expand. Evidence/reports can remain
available independently. Neither paths nor stored presentation supply authority.

Required scientific declarations are not architectural faults. Replicate/design,
matrix semantics, ground-truth choice, tissue/context, namespaces and source
history must remain declared or established by the appropriate authority.
There is, however, a concrete prose-binding gap for ordinary required label
columns. A lightweight eligibility check shows `label_key`,
`reference_label_key` and `ground_truth_label_key` are Registry `(str,)`
scientific declarations with single direct request ports, but the current shared
gate requires numeric accepted types and rejects all three as `unsupported_intent`.
[Eligibility record](/tmp/agent-ua354d-preservation-6h6ec749/label-declaration-eligibility.json)
does not run a Planner or owner. Missing live evaluation evidence is a separate
coverage limit; it is not the cause of that structural exclusion.

## Selection candidates remain unqualified

The [Selection owner contract](m11.4c-explicit-selection.md) and
[threshold profile](../src/agent/tools/data/scatac_selection_profile.py) define:

| Parameter | Present status | Remaining qualification question |
| --- | --- | --- |
| `min_qc_fragment_records` | Required nonnegative integer; no default. **1000** is a candidate. | Scope canonical-record depth, producer/support meaning, QC contigs, assay and depth/population retention. It is not interchangeable with every external mapped-fragment count. |
| `min_tss_enrichment` | Required exact integer/decimal/fraction threshold; no default. **4** is a candidate. | Match annotation/windows, endpoint incidence, strand/deduplication, background, inclusive comparisons and biological retention. External scores/cutoffs are not automatically equivalent. |
| `min_tss_flank_evidence` | Optional; `None` disables. **20** is a candidate additional condition. | Assess low-background instability and retained populations under this exact metric; do not silently enable an optional gate. |
| `max_qc_fragment_records`, `max_nucleosome_signal` | Optional; `None` disables. | No approved general maxima. Depth or nucleosome ratios alone do not establish doublets or purity. |

The retained survey already examined the external references; no fresh literature
survey is needed here. Its flank-20 argument was explicitly an inference:
ArchR's background floor ceases to change its denominator at F≥20, giving an
algebraic relation to Agent's `200*C/(101*F)`. Different counting, strand/site
treatment, incomplete windows, contig scope and rounding still prevent a claim
of full metric equivalence. The [Agent QC contract](m11.4a-qc-resources.md)
explicitly disclaims numerical equivalence to those implementations.

[M11.8](m11.8-biological-closeout.md) used 1000/4/20 as frozen explicit human PBMC
request choices and found bounded descriptive plausibility. It did not calibrate
general defaults. [M14.11](m14.11-real-macaque-qc.md)'s 1/1 minima are permissive
acceptance operands, not macaque policy. UA4b's 2/0.5 demonstrates binding and
verification on a synthetic fixture, not a biological policy.

A narrow future owner qualification can reuse the preserved PBMC QC/evidence
as an anchor, without rerunning raw production. It still needs a frozen assay,
producer, contig/resource/profile applicability statement; assessment of
low-background/undefined-score behavior and depth/population retention; and
independent scientific review with approval of the claimed scope. One descriptive
dataset does not establish portability across species, protocols or sequencing
depths. If approved, existing configuration can supply the existing threshold
arguments with recorded applicability/provenance. Today all three candidates
remain unqualified; explicit clarification already completes first Selection.

## Human EpiZoo: precise resource connection and scientific evidence

Human and mouse inference use **the same original jointly pretrained checkpoint**.
Species selects the original auxiliary resources, normalization constant and
token interval; it does not select a separately trained human model.
[MODEL_CONFIG/HUMAN_CONFIG/MOUSE_CONFIG](../src/agent/tools/models/epizoo.py)
and the checkpoint/device/dtype [cache key](../src/agent/tools/models/epizoo_cache.py)
already implement this. The exact recorded identities are:

| Resource | Established SHA-256 |
| --- | --- |
| Joint `pretrained_EpiZoo.pth` | `6b2d13fdbd54a9b0d56efa5afa81bc4832b813f4eac8662e4d93cc08c9d9b39a` |
| Original `cCRE_frequencies_human.npy` | `8b576fa4fc60a1e2fc77607ffacff2883441def3d8cb8225776b71ad6c00e80b` |
| Original `cCRE_filter_idx_human.csv` | `994d9c3e87208074e695c4c418b28d9587dd8991ad033cf33e62f96ceebc7875` |

The [original interface audit](epizoo_interface.md) records human species ID 0,
1,355,445 raw features, 700,460 retained names, normalization count 8,200,000,
base offset 4 and species offset 0. Existing
[M14 source qualification](../src/agent/tools/models/epizoo_adaptation/resources.py)
loads both species' auxiliary contracts, binds their original frequency/filter
identities and retained vocabulary, and qualifies the same checkpoint/architecture.
The retained
[source bundle](/home/likeyi/program/agent-acceptance/m14.2/qualification-final/source-bundle.json)
has identity `9fa015094881ca168ea281c8234f374e3f704c90a998beb09ec4374404d7982a`.
Its human retained-name digest is
`3bd740e6613a13c6dfd41d986e9f82da67478ccdaefea15f90b503cda73268b8`.
This is reusable source/resource evidence; target adaptation does not itself
execute ordinary human inference.

There is also **actual historical human inference**, independently of adaptation:
[M12 closeout](m12-matrix-level-input-closeout.md) retains M12.2a's canonical C0
control on 32 human PBMC cells. Its
[inference record](/home/likeyi/program/agent-acceptance/m12.2a/pbmc-paired-32-r1/inference.json)
identifies exactly the checkpoint and both human auxiliary hashes above, original
human preprocessing, batch 4, seed 0, float32 CUDA/AMP and finite `(32,512)` output.
Actual model token inputs and cell order were checked. The C0 control uses exact
canonical counts; this does not approve the experiment's B/P transformations or
the rejected peak-projection capability. It is lower-level bounded scientific
evidence, not a current pinned registered-Web RunStore authority.

The read-only [human evidence review](/tmp/agent-ua354d-human-default-evidence-8kzmxj77/human-default-review.json)
matches current small auxiliary hashes against both the M14 bundle and M12 C0;
retained small records/arrays match their original audit. Model/species constants
and the six existing scientific API calls are unchanged. The large checkpoint
is only statted. Thus a broad assertion of **no human direct-inference evidence**
would overlook M12. Earlier UA3.2/UA4 records remain intact; their missing current
interactive acceptance must not be read as missing joint-model qualification.

The actual failure precedes science:
[select_epizoo_resource](../src/agent/application/local_resources.py) looks for
exactly one configured default matching the explicit species. The unchanged
operator catalog contains only `mouse-recorded-qualified`; for human the matching
set is empty, so it returns `EPIZOO_RESOURCE_REQUIRED`. The existing configuration
type already supports a human entry with that same checkpoint and the two human
auxiliary pins. No new selector, inferred species, checkpoint, retraining or
normalization algorithm is required.

| Question | Evidence-based classification |
| --- | --- |
| Human checkpoint/model missing or separately unqualified? | **No.** Same original verified joint model; both vocabularies exist. |
| Original human auxiliary files/identities missing? | **No.** Audited, source-qualified, used by real C0 and current hashes match. |
| Human Agent resource entry/default mapping missing? | **Yes.** Concrete operator/catalog connection gap. |
| Known genuine incompatibility of those original human resources? | **Not established.** No owner-contract violation demonstrated by this inspection. |
| Compatibility of any arbitrary human H5AD? | Still checked per input: sparse count-like values, full dimension and exact retained names/order. A human declaration or same checkpoint cannot prove it. |
| Current pinned registered-human interaction + real science + browser pass? | **Not yet demonstrated.** Separate integration/UA3.6 evidence, not redundant model qualification. |

The historical observation of 16 human DF values above 8,200,000 remains a
provenance question. The original interface explicitly says not to reject those
files for it. The existing owner requires finite, nonnegative correctly sized DF;
native TF-IDF uses `log1p(N/(df+1))`, and C0 used these same resources/settings.
That observation establishes neither runtime incompatibility nor an approved
normalization change. It is not an invented prerequisite for reconnecting the
existing reviewed resources.

All existing consumption checks must remain: exact bytes/snapshots, strict joint
checkpoint loading and cached checkpoint identity, human auxiliary/order checks,
fixed inference settings, requested device, finite float32 output, cell order and
matching owner provenance. UA4c's mouse admission does not prove those resources
were consumed in its withheld run. Historical real mouse/human execution and
current deterministic consumption-safeguard acceptance remain separate evidence.

## Ranked next work — one selected target

### 1. Preferred: connect original human EpiZoo resource/default mapping

**User limitation:** an otherwise valid declared-human embedding request cannot
resolve the current deployment's EpiZoo resource choice. This blocks the EpiZoo
entry to embedding/neighbors/Leiden/UMAP and human reference/query workflows that
need embedding. The scientific owner and selector already exist; the missing
piece is the operator catalog row, not an algorithm or a human checkpoint.

**Smallest next task:** designate one reviewed `QualifiedEpiZooResource` for
`species=human`, using the existing checkpoint path and exact three hashes above,
with a qualification description citing the original resource/source evidence
and canonical M12 C0 scope. Make it the unique human default through the existing
`epizoo_resources` configuration boundary. Preserve the mouse entry, planning
profile, explicit-choice priority and unrelated operator fields. Production
components `QualifiedEpiZooResource`, `select_epizoo_resource`, H5AD composition,
species continuation and `epizoo_embed_cells` already accept this representation;
**no production source change is presently justified**. A future configuration
change must be explicit operator admission, not automatic catalog discovery.

**Necessary authority:** existing joint-checkpoint and original human auxiliary
qualification, the project owner's joint-foundation policy, and an accurate
operator designation of this exact resource tuple. M12's experimental control
is supporting scientific evidence, not authority fabricated for a new Run.
No repeated model validation, adaptation or retraining is a prerequisite.

**Focused acceptance for that next task:**

- Reconcile the small human auxiliary pins with retained qualification and
  current files; reuse the established large-checkpoint identity rather than
  reload it merely to register a mapping.
- Existing configuration/selection tests must cover explicit human choice,
  unique human default, missing/ambiguous defaults, species conflicts and changed
  pins, while preserving the mouse/default and default planning profile.
- A deterministic human species continuation must retain the original objective,
  source/device and select the exact human pins through ordinary compilation and
  Application resource admission. Missing-resource and identity failures must
  still fail closed; no semantic conversion or fallback is allowed.
- Keep the existing owner checks and consumed-resource provenance tests active.
  The UA4b A2 mouse fixture declared human is not a valid positive human inference
  fixture: adding a mapping must not make its mouse feature axis pass human checks.
- Record configuration/admission acceptance separately from any later approved
  current human pinned execution. UA3.6 can exercise that existing owner on a
  genuine compatible bounded human input with live semantics/browser; it need
  not requalify the joint foundation or rerun the M12 projection experiment.

This is safe to prepare as the next configuration integration because resource
identity, existing human scientific execution and the shared owner already exist.
This assessment does not activate the entry or claim the current Web path passed.

### 2. Shared exact string declarations for the three required label keys

**User limitation:** complete structured inputs can execute clustering evaluation,
reference label transfer and annotation evaluation, but an explicit label-column
name in chat cannot fill their required string declaration through the numerical
binder. This is an interaction gap affecting three registered tools.

**Smallest repair:** extend existing typed declaration projection/admission and
compiler-grounded pending continuation for reviewed scalar text contracts,
starting with `label_key`, `reference_label_key`, `ground_truth_label_key`.
The responsible components are Registry argument/port metadata,
`turn_decisions.py` and `dialogue_execution.py`, with existing persisted binding
validation. Reuse exact spans, exact strings, singleton actual-consumer scope,
atomic correction and owner checks; add no per-tool language grammar.

Existing metadata can isolate these three required `(str,)`, request-only,
nonartifact, reference/ground-truth declarations with unique single-member ports.
Eligibility requires explicit contract review; `scientific_parameter=True` or
`allow_step_output_ref=True` alone is insufficient. Broad string admission would
also capture device, namespaces and experimental design operands. Lists/dicts,
resource paths/digests, source history and broader scientific choices retain
their owners. The scientific owners already validate column presence, labels,
source roles and exact cell order.

**Focused acceptance:** supplied/omitted keys, exact quoted names/spans, valid
original-task continuation, corrections, stale source/generation, duplicate
consumers, nonexistent columns and preserved ground-truth/reference/query roles.
No science while a required key remains missing, no column inference or default,
and no scientific publication for invalid input. Reuse complete trusted input
sets for the first witnesses; general uploaded evaluator/multi-H5AD composition
is a separate boundary. Safe bounded software work, but broader than priority 1.

### 3. Connect one source-specific approved raw companion

**User limitation:** new uploaded fragments/BAM/FASTQ sources can be inspected or
registered, but this deployment has no activated scientific companion for their
production/QC/Selection/matrix routes. Across the three entry routes, this affects
seven RP tools even though their scientific owners and input composition exist.

**Smallest repair:** operator provision one applicable source-specific companion
through the existing reference/library/QC APIs and Web InputSet configuration,
starting with one reviewed route/source. `WebConfiguration` and
`LocalResourceAdmission` already own the connection; producer, library and
reference owners own its qualification. No new workflow/default engine is needed.

**Acceptance:** exact registered source membership/pins, valid source-bound
context/reference/runtime, explicit profile/namespace/history where applicable,
typed prerequisite errors, unchanged source on retry/restart and no science from
upload alone. Selection minima remain explicit. An old FASTQ/BAM context cannot
be reassigned to uploaded copies merely because content matches. Preparation is
safe once the exact source declarations/context exist; missing corrected-barcode
or library authority cannot be repaired by configuration guessing. This is
operational provisioning, not a justified generic production-code fix.

Selection calibration and processed-output authority/export expansion remain
documented owner work, rather than additional ranked implementation targets.
Their missing evidence/contracts should not be hidden behind a transport or
default-value patch.

## UA3.5 readiness and UA3.6 boundary

The common mechanisms now have bounded acceptance: typed intent admission;
Registry-grounded scoped scalar binding; H5AD species pending/resumption;
compiler-authored missing-input handoff; partial answers and atomic corrections;
original input/resource identity; owner defaults and accepted provenance;
actual-plan output selection with exact offered pairs and unique names; durable
clarification/retry/refresh behavior. Live tests demonstrate Leiden and Selection
execution and mouse species/multi-step admission. They do not live-qualify all
23 tools or broad language/biological reliability.

Practical completion is still limited by deployment connections (especially the
existing human EpiZoo tuple and raw companions), explicit text/declaration/input
composition coverage, and unsupported generic cross-turn authority/export for
processed outputs. These are distinct from unapproved empirical policies or
incompatible scientific inputs. Human catalog mapping is the smallest concrete
next repair; the discovered historical human evidence strengthens that choice.

A scoped UA3.5 closeout can retain explicit required thresholds, source-specific
resources, experimental design/tissue declarations, absent statistical cell
calling, unsupported peak projection and unqualified production adaptation as
documented boundaries. It need not invent defaults or individually benchmark
23 tools. Broad downstream cross-turn reuse and universal upload composition
cannot be silently counted as accepted; their inclusion in closeout requires an
explicit scope decision and the appropriate existing owners. No closeout is made
by this assessment.

UA3.6 retains the combined live-provider + real-science + native-browser
acceptance, current pinned foundation consumption and feature compatibility on
actual inputs, broader continuation/backend-language reliability and biological
interpretation. Existing historical science is reused rather than erased or
rerun merely to establish that the joint model supports human and mouse.

UA4c's first full Application/Web run had one unfinished-BAM fixture failure
before detailed planning/output selection, with an uncaptured underlying
exception. The isolated 33-test pass and complete passing rerun establish no
root cause. It remains **unresolved**, without classification as caused by output
selection, definitively flaky or pre-existing. If it recurs in relevant future
regression, capture the actual exception and responsible boundary before proposing
a repair. No BAM investigation or new regression is performed here.

Final preservation and document checks are recorded in
[assessment audit](/tmp/agent-ua354d-preservation-6h6ec749/final-audit.json).
Only this report is added to the repository. The chosen next task remains a
proposal; production, configuration, providers and science are untouched.
