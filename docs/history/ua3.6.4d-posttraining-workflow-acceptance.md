# UA3.6.4d — Shared genome ownership and native post-training

Date: 2026-10-14. Work is unstaged; no commit or push. This record supplements,
without rewriting, [Work A](ua3.6.4d-reference-supplementation-acceptance.md).
The mandatory shared-genome correction and a new, bounded native training →
inference → analysis run pass. Overall workflow acceptance is **PARTIAL**:
presentation needed explicit recovery, the live-model plan failed admission,
and omitted-budget guidance and general input-representation support remain
incomplete. These limits are not hidden by the successful scripted science.

## Baseline and evidence locations

Starting branch `main`, HEAD `ce5b53d7f716b777fc5333ea84e4d868ccf0da71`.
The inherited target-inference and Work A changes were inventoried before edits:
1,644 repository files, including the existing evaluation files. The snapshot,
inherited patch, isolated operator configuration, native Firefox harness,
provider ledger, test commands and scientific event logs are outside Git at:

`/home/likeyi/program/agent-acceptance/ua3.6.4d-posttraining` (**B**).

Historical Work A and earlier UA3.6.4d reports remain unchanged. No biological
matrix, genome or checkpoint is added to Git. EpiZoo's scientific implementation
and `src/agent/tools/models/epizoo.py` are unchanged.

## Corrected ownership

Work A offered a combined `regulatory_reference` and required the H5AD's exact
ordered features in `reference_supplementation.select`. That remains correct for
that combined resource, but does not establish a reusable genome abstraction.

`application/shared_genome.py` adds an explicit genome-only approval using the
existing local catalog and optional qualification metadata. Its version is
`qualified-genome.v1`; there is no additional catalog, database or scientific
verification engine. The identity includes species/assembly, exact FASTA/FAI
content, contig geometry and source provenance. It excludes H5AD, BED, ordered
peak vocabulary and model identity. Code and file pins preserve qualification.

The operator-only `shared_genome.promote(...)` checks an existing owner-qualified
combined record, captures its evidence identity, and requires new explicit
operator/provenance/reuse declarations. Historical promotions are not silently
reinterpreted. The new record points to existing FASTA/FAI bytes and survives the
loss of the originating dataset BED. There is no browser publication endpoint.

Selecting a shared genome derives a private, exact, ordered feature reference for
the captured H5AD. M14.1's unchanged builder checks its coordinates and publishes
its unchanged feature-reference contract. A persisted private binding records the
selected genome handle for deterministic continuation replay. Different features
produce different feature references; no alias repair, alignment, dropping,
liftOver or curated-cCRE inference occurs in this product binding.

This deliberately retains full M14.1 verification when creating a new feature
binding. It does not yet provide a standalone owner proof that skips genome
scanning across new bindings. Existing matrix/adaptation verification can also
scan the genome. Catalog reopening itself only checks pinned identity/content;
it does not repeat scientific reconstruction.

## Genuine genome reuse

The already uploaded Work A FASTA/FAI is reused; there is no second 1.7 GB upload
or genome copy. Its provenance remains M14.5 operator-supplied, not independently
proven publisher/redistribution permission. Current user authorization covers
this isolated same-operator deployment, not public redistribution.

- FASTA: 1,712,826,111 bytes; SHA-256 `9d45886b4c686f5fef63e53f14a02277c0c4467d46982dd8dee742420f84661a`.
- FAI: 79,526 bytes; SHA-256 `8be439d7b9c529203a0adf64ca9183b6f72b66d3f2c0bd8adc0924e8e56818aa`.
- Species **Danio rerio**, taxonomy **7955**, assembly **danRer11**; 1,923 contigs.
- Genome identity: `31fbf90468e9a31a3775729f6305b3e7125a76978ff8900161d7bcf30abd5c91`.

`B/genuine-genome-bindings.json` records two distinct four-region subsets of the
real data under that same genome. Their feature-reference identities are
`12fbee522aab16e7479f2b218ab1ee884781d7a2328dbb26e4e98281def3e730`
and `1dd41598a5d599ad2f44455e01325bc5bb8c55b6a1d03a4d3d40325d5da4bb8c`.
Both retain exactly the same genome object and paths; no FASTA is copied into the
fixture workspace. Fresh catalog construction recovers the genome. This is
programmatic owner qualification with explicit fixture pending contexts, not an
accepted scientific Run or biological validation. The Web execution below uses
another real feature set through actual Session continuation.

Focused negative checks cover different assembly, unknown contig, out-of-bounds
coordinates, changed genome bytes, stale/source-substituted requests, private
scope, catalog corruption, and missing sharing review. Existing Work A combined
resource behavior and historical replay remain supported.

## Native configuration and historical compatibility

`epizoo-adaptation-inputs.v2` selects `native.v2`; its source/SEAM bundles and
species/assembly remain pinned. A null reference identity explicitly delegates
the particular feature reference to the exact adopted matrix's qualified owner.
It does not remove species, assembly, measurement or model-vocabulary checks.
Historical v1 specifications, profile hashes and fixed rules are retained.

The new scientific method is `epizoo-native-posttraining.v2`. Its configuration
resolver instantiates the pinned `EpiZooXPostTrainConfig`, records its resolved
trainer fields, and applies only explicit `max_steps`, `learning_rate` and
`training_batch_size`. The existing scalar declaration/Registry/compiler path
binds these values; there is no species-specific text parser or named profile per
value. Native optimization, scheduling, SR/CCA, parameter transfer and training
remain in EpiZoo. Batch sizes 1–4 retain the existing qualified 24 GiB loader
bound. Seed, sequence loader and other fixed execution settings remain explicit.

The native trainer default is **500,000 attempts**, learning rate **5e-5**,
logging every **500**, saving every **4,000**, and retaining **5** periodic
checkpoints. These come from the native dataclass, not an Agent copy. Native-v2
execution requires an explicit positive step budget; omission fails closed and
cannot silently start the 500,000-step default. The complete interactive estimate
and confirmation experience for an omitted budget is not qualified here.
Explicit large schedules and arbitrary batch/matrix feasibility are not claimed
qualified by this small acceptance.

Fully resolved configuration remains in the exact model manifest arguments and
therefore its receipt, execution identity and accepted authority. Native model
evidence/report projection additionally exposes the safe effective configuration,
method, attempt/update counts and an explicit unassessed biological-performance
label. Historical report facts remain readable without these new fields.

Resource bundles pin Agent adapter bytes as well as EpiZoo. Consequently new
bundles were qualified using the original resource owner and unchanged source
weights, SEAM weights/configuration and EpiZoo revision. Old bundles were not
edited or relabelled. Historical training recovery still has its original strict
code/runtime requirements; this work does not migrate that authority. The
existing narrower accepted-model inference consumer continues to admit the
real M14.5 checkpoint with zero training reconstruction (`B/historical-target-check.json`).

## Scientific execution scope and attempt history

Before GPU execution: RTX 4090, 24,564 MiB total, 23,867 MiB free; approximately
15.6 TB free disk. No dependency installation, GPU scheduler or training database.
CPU thread limits and all scientific settings are recorded in the isolated run.
The technical schedule is **10 attempts, learning rate 1e-5, training batch 1**;
sequence batch 4, seed 0. The 500,000-step default is never used.

The first browser harness quoted an entire multi-number sentence for each
parameter. Exact grounding rejected it before science. The corrected harness
quotes each unique parameter clause. A later real admission exposed an existing
H5AD guard that required a pre-existing model even when the plan explicitly bound
a newly trained model. The fix accepts only a paired `manifest_path` /
`manifest_sha256` dependency from `adapt_epizoo_species`, then checks its exact
verified result at acceptance. Unpaired, wrong-tool, failed, unverified,
substituted and ambiguous bindings fail. It does not select models by species,
step order or first match. These failures and their Sessions remain in B.

The first scientific attempt used 16 real cells with all 466,307 historical
features. It was explicitly cancelled during SEAM after 743.44 seconds. No native
training call or optimizer update occurred; the browser reports **Cancelled**,
with no accepted model or Revision. Its performance difference from the older
M14.5 observation is not established as a scientific algorithm defect.

That attempt also exposed native pyfaidx's sibling-index creation: it added a
79,526-byte `.fa.fai` alongside the historical Work A upload. No pre-existing
FASTA/FAI bytes changed. After cancellation, this newly created file was moved to
`B/native-created-work-a-sidecar.fai`; this transient directory write is disclosed,
not hidden behind a blanket no-writes claim. Native-v2 now calls the same EpiZoo
sequence extractor with a temporary link to the shared genome and a small local
copy of its qualified index. Native indexing can only write inside that temporary
namespace; the genome is not copied. Production and independent verification use
this same placement adapter. A negative test checks original directory bytes and
complete scratch cleanup.

The subsequent explicit technical input is **16 × 17,922 int64 CSR**, retaining
all **19,959** nonzeros of those 16 cells and the original feature order. The
features are exactly those observed in the selected cells. This is a separately
labelled acceptance submatrix, not product-side feature filtering or a change to
native model compatibility. The original full-vocabulary input is preserved.
`B/technical-subset.json` records every selected feature index. No float-to-int
product conversion is claimed: the current neutral adoption contract still
requires canonical int64 CSR. Supporting general lossless representations would
need explicit source/output conversion provenance and verification at that owner;
this remains a user-facing limitation.

## Ownership and delivery boundaries

New checkpoints, corpus DF, target sequence embeddings, input matrices and
analysis artifacts remain task outputs under the existing Session/Revision and
RunStore. No new model is registered in a global InputSet or shared reference
catalog. The shared genome approval cannot authorize a model vocabulary mismatch.

The existing deployment has no authenticated tenant boundary. Session-scoped
presentation is not a claim of secure cross-user filesystem or account isolation.
General model publication remains outside scope. Existing compact analysis,
embedding/graph/UMAP exports are retained. Checkpoint HTTP download is not added
or promised; model publication paths and digests are internal scientific records.

## Software validation

The affected selection completed **958 passed, 3 skipped, 3 warnings in 182.79 s**,
exit 0, with all `RUN_*` gates removed, no test exclusions and all 512 frozen
source/test/static files unchanged during execution. The exact command is in
`B/focused-final-result.json`. This includes resource/upload/continuation,
application/session boundaries, historical target and human/mouse inference,
reference owners, adaptation lifecycle/verifiers, report/evidence and compact
analysis delivery.

Two inherited failures were corrected to match established contracts: assembly
is now a supported target declaration (with a new positive test that it confers
no model authority), and interpreter wire declarations use exact `evidence`
quotes rather than supplied offsets. Neither scientific assertion was weakened.

After the native index-location correction, the affected owner/configuration,
public integration and accepted-target selection passes **156 tests in 13.99 s**.
The final shared-reference tests pass **15 tests in 5.20 s**, including independent
feature identity, invalid coordinates and a missing originating BED. No complete
repository regression is claimed.

The genuine execution subsequently exposed a Scanpy neighbors operation changing
the already-loaded PyTorch CPU thread count from 4 to 32. All six scientific steps
passed, but later model-evidence verification correctly rejected the changed
runtime. A bounded before/after diagnostic establishes the cause. The neighbors
wrapper now preserves and restores the existing thread setting around the Scanpy
import/operation, including failures; it does not import PyTorch if absent or
change the neighbors computation. Embeddings, graph arrays, labels and UMAP are
exactly equal across the diagnostic's original and corrected outputs.
After this correction, **110 analysis/publication/delivery tests pass in 12.83 s**,
including new success/failure restoration checks. Evidence is in
`B/thread-diagnostic*.json`, `B/thread-result-invariance.json` and
`B/final-neighbors-tests.log`. The 958-test run predates these last index/thread
changes; the later focused results qualify those changes, rather than claiming
a final full regression that was not run.

## New training and same-plan downstream execution

Native Firefox uploaded the labelled real submatrix, submitted the training
request, selected the offered shared genome in a normal follow-up and executed
the normal compiled/preflighted plan. The interpreter and Planner were scripted
for this scientific acceptance; every scientific production and verifier was real.
No acceptance stop was installed for this scripted execution.

```text
adopt_scATAC_cell_by_features → adapt_epizoo_species → epizoo_embed_cells
  → build_cell_neighbors → cluster_cells → compute_cell_umap
```

Session: `1240e849-68a2-4a75-8a4b-d9d015a65650`.
Execution turn: `e054a871-1a39-49d2-a1a0-a999ba3afdd3`.
Run: `turn-6941ac2c2fed589dfcdd0307041f8e4c40cc75245db4a8b85b62e43a1b4eee6e:run`.
Accepted Revision: `0a7e5d9feebfa0f60d4decd7baf68ddc91bd7de13a768c8cdc8b641765f7ca87`;
generation 1.

The full effective native trainer configuration was:

```json
{
  "mode": "sr_cca",
  "max_steps": 10,
  "lr": 0.00001,
  "weight_decay": 0.01,
  "warmup_steps": 1000,
  "epoch_decay": 0.9,
  "sr_weight": 1.0,
  "cca_weight": 1.0,
  "use_amp": true,
  "grad_clip": null,
  "freeze_seq_emb": true,
  "max_cca_metric_samples": 10000,
  "save_steps": 4000,
  "log_steps": 500,
  "keep_last": 5,
  "log_file": "training_log.csv"
}
```

Placement is the managed artifact output directory on `cuda:0`. Training batch
is 1, sequence batch 4, workers 0, seed 0, model dtype float32, autocast float16,
maximum cell sentence length 8,192 and sequence length 512. The existing
512-dimensional, 30-layer architecture and preprocessing remain unchanged;
sequence embeddings are frozen. `B/model-observations.json` preserves the entire
effective profile, parameter scopes and runtime, including PyTorch 2.7.1+cu128,
CUDA 12.8, CPU threads 4 and interop threads 24. No resume, hidden truncation
change or schedule adjustment occurred.

| Measurement | Observed result |
| --- | --- |
| Native attempts / applied updates / AMP skips | **10 / 10 / 0** |
| Finite-loss attempts | **10** |
| Final total / SR / CCA loss | 4.5535502434 / 3.4608697891 / 1.0926805735 |
| SEAM / native trainer / target inference | 126.41 s / 19.38 s / 2.77 s |
| Browser continuation to terminal scientific result | 327.27 s, including reference work; initial evidence composition failed as described below |
| Peak GPU allocated / reserved | 6,676,896,768 / 7,090,470,912 bytes |
| Target inference | **16 × 512**, finite float32 embeddings, exact ordered cells |
| Leiden / UMAP | 1 cluster at resolution 0.5 / **16 × 2** finite coordinates |

These ten steps are a technical execution test, not convergence, calibrated
clustering, cell-type accuracy or cross-species biological generalization. Local
GPU execution has no monetary billing record; elapsed time and memory above are
measured. The earlier cancelled SEAM attempt is additional resource use, not part
of the successful run's timing.

| Identity | SHA-256 |
| --- | --- |
| New checkpoint | `716b80088fb457559401b7a7dde3711439e5a843a106b31ca32e9781ab9e5777` |
| New model manifest | `49aaac55da9189ce55039f21dcfbb214991005af786a4ec3672c0ee5a6ccbf77` |
| Method/profile | `dd40bcb0364936b9db57f757a298ce99c4d6b904f857980ce0da5c50185113d2` |
| Dataset feature reference | `92d9e55082a35852d17a2aef2907a38f6314a9a0c5c9e6796925e62cf2669a03` |
| Ordered model vocabulary | `fef49956379c596be63e08aa8a195ec7c234efa8f2b2bd8530c09e56e56002a8` |
| Adaptation specification | `2e524119b9750a8b2db564efaa397b9d07ebd3a7a6e20279e346be90b41196d4` |
| Source-model bundle | `59b87c0b72bd86d8e0c2ecad27bc8e284bde207a432770ffb1ea93b069421a21` |
| SEAM bundle | `dd1a12e798c2f182e8760a51efe07f4554245e514bb62d2c9a3eeaf9a87a93ff` |

The checkpoint is newly trained here, not the historical M14.5 checkpoint.
Explicit paired step-output bindings pass this exact model to target inference;
there is no global model registration or inferred model choice. Production SEAM,
training and target inference each ran once for this successful attempt; mapping
ran zero times under `de_novo`. The five outer target-verification calls include
one deep verification (36.65 s) and subsequent operation-proof reuse (about
0.02–0.03 s), not five reconstructions. Reference build/deep FASTA verification
each occurred five times across the current owners. This remaining repeated
reference scan is disclosed; it is not a new duplicate verification engine.

## Presentation, downloads and fresh-process recovery

The first terminal browser display truthfully reported `APP_EVIDENCE_FAILED`
after all scientific steps succeeded. The thread-setting fix above was verified
on a small diagnostic without retraining. A fresh-process application resume and
explicit facade `recover_turn(..., complete_presentation=True)` completed evidence
and accepted the Revision with provider construction, training, SEAM, mapping,
corpus reconstruction and reference building guarded against execution. This is
presentation recovery of completed science, not interrupted-training continuation.
The initial diagnostic script also had a wrong JSON-key assertion; both diagnostic
attempts and their logs are retained.

Reopened Firefox shows the accepted model's effective configuration, attempt and
update counts, report, exact Revision, scientific downloads and UMAP. The original
captured turn error remains visible in its history alongside the recovered
completed result; it was not erased or presented as an error-free first run.
This is a remaining presentation limitation. `B/presentation-recovery.json`,
`B/native-science.json` and screenshots preserve both states.

All three actual HTTP downloads were checked using browser SHA-256 and read back
as H5AD. The final read-back results are in `B/final-restart.json`:

| Download | Contents | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `analysis.h5ad` | Compact 16-cell object, `X_epizoo` 16 × 512, `leiden`, `X_umap` 16 × 2 | 53,712 | `122f23568db2a820f6f50e461cd4fce4f511ba6200befcd69b48d805c6da61fb` |
| `matrix.h5ad` | Accepted 16 × 17,922 regulatory-feature matrix | 1,067,628 | `dc8d5e02910a164e680322bb2ba25c48c76ac116db462acb34ce7442971c19ce` |
| `neighbors.h5ad` | Compact 16-cell embedding plus connectivities/distances | 83,440 | `5291f4246c0f70cbbcedca3c86e03cbefe70670212b9bedab0f5f5f664129ff4` |

Two accepted PNG handles render the same 1,600 × 1,200 UMAP bytes in this result;
both return HTTP 200, 29,453 bytes and SHA-256
`37a108cd3708ccfef76071af3bf6b4cf3d9097f3b814cf96d81ba6733b2c52cb`.
They are not two independent scientific figures. Native screenshots include
`B/final-figure-2.png`. Checkpoint download is unqualified and was not exposed.

After the live attempt below, a new server in a strict read-only harness reopened
the Session, six captured conversation texts, generation 1 and the same Revision.
It displayed both PNG handles, evidence and downloads and refreshed successfully.
All **102 workspace files** retained identical hashes; provider construction and
instrumented scientific calls were **zero**. The reusable genome catalog also
survives fresh construction as tested separately. No automatic training restart,
SEAM replay or scientific-authority manufacture occurred.

## Live OpenRouter result

One continuous native-browser turn used **`openai/gpt-5.6-sol`**, the existing
durable ledger, price caps, $3 task reservation cap, zero SDK retries and no
fallback. Credential presence was checked only as a boolean. Unlike Work A's
transport failure, this attempt received four real responses: interpreter,
capability selector, initial Planner and the existing bounded Planner repair.
There was no manual semantic retry or second user request for a favorable result.

The interpreter selected new adaptation and correctly grounded assembly,
10 steps, learning rate 1e-5 and batch 1. It did not select the typed reference
prerequisite. Both Planner candidates included adoption, adaptation and downstream
analysis, but selected an incomplete request source bundle for adoption.
The compiler returned `MISSING_REQUEST_SOURCE_MEMBER`; its repair diagnostic
required `select_complete_request_source_bundle`. The second candidate repeated
that source choice. Normal admission rejected it before execution. No live
training, SEAM or inference ran, no new Revision was created, and the accepted
scripted result remained active. The acceptance-only guard against a second
training run was not reached. No scientific contract was weakened to admit this
plan and no species-specific language repair was added.

The harness intended a fresh Session, but its new-session wait returned during
page hydration and the request actually reached the existing accepted Session
`1240e849-68a2-4a75-8a4b-d9d015a65650`, live turn
`64d54c01-eba3-4277-b7f0-57d6dbb46f89`. The raw ledger and browser snapshot agree
on that identity. Consequently this is a failed live continuation attempt with
existing-result context, **not qualification of a fresh-session live flow**.
The exact cause is bounded to the observed provider choices/admission failure;
the harness context difference prevents stronger conclusions about intended
fresh-session quality. No provider rerun was made to hide either limitation.

| Provider accounting | Amount |
| --- | ---: |
| Returned calls | 4 |
| Prompt tokens / completion tokens | 20,362 / 6,324 |
| Reserved upper bound, not refunded | **$0.392612** |
| Sum of returned provider cost fields | **$0.1138535** |

These returned cost fields are provider observations, not an independently audited
invoice. Request/response bodies and the ledger are in `B/provider/`; final UI
state is in `B/native-live.json` and `B/live-outcome.png`. The workflow is **PARTIAL /
not qualified live end to end**, not transport-blocked and not a successful live
resource-to-training chain.

## Exact change inventory and preservation

Relative to the captured starting tree, 15 existing files changed:

- `src/agent/application/reference_supplementation.py`
- `src/agent/application/local_resources.py`
- `src/agent/orchestration/species_adaptation_registry.py`
- `src/agent/report/analysis_report.py`
- `src/agent/report/evidence.py`
- `src/agent/tools/analysis/embedding_analysis.py`
- `src/agent/tools/models/epizoo_adaptation/backend.py`
- `src/agent/tools/models/epizoo_adaptation/contract.py`
- `src/agent/tools/models/epizoo_adaptation/publication.py`
- `src/agent/tools/models/epizoo_adaptation/resources.py`
- `src/agent/tools/models/species_adaptation.py`
- `tests/application/test_h5ad_composition.py`
- `tests/application/test_turn_decisions.py`
- `tests/tools/analysis/test_embedding_analysis.py`
- `tests/web/test_reference_supplementation.py`

Five files were added:

- `src/agent/application/shared_genome.py`
- `src/agent/tools/models/native_posttraining.py`
- `tests/application/test_new_target_composition.py`
- `tests/epizoo_adaptation/test_native_configuration.py`
- This acceptance report.

`B/final-preservation.json` records unchanged hashes for all **981 evaluation
files**, the other **1,629 captured repository files**, all **35 Work A workspace
files**, and **14 selected historical scientific/resource files**. All **436**
files in the preceding UA3.6.4d acceptance tree retain original size/mtime.
This is a bounded preservation audit, not a new byte audit of every historical
RunStore or checkpoint. The transient new FAI described above is now absent
from Work A. Existing historical reports are unchanged. EpiZoo remains clean at
`029cd631d0f9806a646c4a3b42ce10b958f2b67f`; its model algorithms and Agent's
`tools/models/epizoo.py` were not edited. The registered inventory remains 23.

No production operator configuration or default was changed. The isolated
server/Firefox processes were stopped; no GPU compute process remained. HEAD
and branch are unchanged, the index is empty, and `git diff --check` passes.
No commit or push occurred.

## Scope assessment and remaining work

| Scope | Assessment |
| --- | --- |
| Genome identity independent of H5AD peak vocabulary; real distinct bindings; no genome copy | **PASS** |
| Existing Work A supplementation, explicit sharing review and private feature/model ownership | **PASS** within the existing operator-managed scope |
| Native explicit parameter overrides, actual new checkpoint and same-plan target inference/Leiden/UMAP | **PASS**, bounded genuine data and scripted provider |
| Evidence, downloads, UMAP and fresh-process zero-science recovery | **PASS after explicit presentation recovery**; historical error remains visible |
| Focused software and historical checks | **PASS** for the recorded selections; no full repository regression |
| Live-model workflow | **PARTIAL**: parameter interpretation succeeded, source admission failed; fresh-session scope invalidated by harness wait |
| Omitted-budget default explanation, cost/resource estimate and typed confirmation | **INCOMPLETE**; currently fails closed, cannot start 500,000 steps implicitly |
| General lossless noncanonical H5AD representations | **INCOMPLETE**; canonical int64 CSR remains required |
| Checkpoint HTTP delivery, authenticated cross-user isolation, global model publication | **NOT IMPLEMENTED / NOT CLAIMED** |
| Full-vocabulary runtime, biological convergence and generalization | **NOT QUALIFIED** |

The ordinary successful path currently needs the operator's qualified source/SEAM
bundles and v2 adaptation specification plus explicit species/assembly, strategy,
measurement semantics and a step budget. The researcher uses the existing InputSet
and uploads/answers; no internal JSON is requested in the browser. This does not
establish automatic acquisition of every new-species prerequisite from prose.

The smallest remaining work is typed omitted-budget guidance/admission using the
existing prerequisite path, owner-qualified lossless source conversion if broader
storage layouts are required, and separate qualification of fresh-session live
resource selection after fixing the harness wait. Those are not implemented by
inventing authority, silently changing data or adding a new workflow/security
platform. This bounded implementation and its disclosed partial acceptance stop
here for review, with no further training, commit or push.
