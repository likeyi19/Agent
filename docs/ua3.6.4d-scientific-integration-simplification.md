# Work Unit D — EpiZoo adaptation integration simplification

2026-10-14 (Asia/Shanghai). **Bounded review; no production or test changes.**
Existing owners already provide the qualified adaptation, native Post-training,
reference supplementation, target inference and delivery paths. This unit found
no demonstrated local defect requiring another implementation. All four UA3.6.4
product scenarios remain **PARTIAL**; successful scripted science does not qualify
the complete live-model researcher workflow.

## Baseline and preservation

HEAD, `main` and local `origin/main` are
`bc9963840394a5df802857d7ce09794d3e990379`. The index starts empty. The inherited
tracked changes are C's clarification text in `application/responses.py` and its
HTTP regression in `tests/annotation_integration/test_external_entry.py`.
Untracked B/C reports and `evals/` are preserved. No fetch, stage, commit or push
occurs. Evidence is `/tmp/agent-adaptation-simplification-v1_4ok93` (**R**).

`before.json` hashes all 676 tracked files at entry, including C's changes, the
B/C reports and operator configuration, and records names/sizes/nanosecond mtimes
for 1,274 eval files. Before testing, `protected-before.json` records 200 files in
the accepted M14.4/M14.5 application trees and UA3.6.4d Post-training workspace;
161 files at most 2 MiB also receive hashes. Larger files, including checkpoints,
receive metadata checks, not another multi-gigabyte content audit. All 77 tracked
native EpiZoo files are hashed; that repository is clean at
`029cd631d0f9806a646c4a3b42ce10b958f2b67f`.

Current behavior is read against the [workflow](ua3.6.4d-epizoo-cross-species-user-workflow.md),
[reference supplementation](ua3.6.4d-reference-supplementation-acceptance.md),
[native Post-training](ua3.6.4d-posttraining-workflow-acceptance.md),
[default/delivery closeout](ua3.6.4d-final-closeout.md) and
[final checkpoint review](ua3.6.4d-commit-review.md). Later records supersede the
earlier mandatory-step-budget and 2 GiB download limitations. Historical records
are not rewritten.

## What a researcher can supply

The ordinary supported path selects a registered H5AD and reviewed scientific
InputSet, supplies the necessary declarations, and requests the intended strategy
and downstream results. Missing reference resources use the existing typed
resource prerequisite. Agent derives the ordered feature reference and binds
internal identities; users need not manufacture manifests, hashes or authority.
An arbitrary H5AD plus FASTA alone does not supply pretrained/SEAM resources or
qualify every strategy.

| Operation | Required data and declarations | Reusable resources and Agent work |
| --- | --- | --- |
| New `de_novo` adaptation / native Post-training | Exact neutral matrix; declared scientific name/taxonomy, assembly, measurement semantics and strategy. External adoption currently requires canonical **int64 CSR**, unique exact axes and coordinate features. Training requires accessible and inaccessible features in each cell. | Qualified source checkpoint/configuration/retained vocabulary, SEAM bundle, native code/runtime and compatible target FASTA/FAI. Agent derives exact ordered BED/reference from H5AD coordinates and invokes existing adoption/training owners. No chain is required. |
| `mapped_reference` | Qualified fragment-count target matrix and explicit target-to-source mapping specification. | The above resources plus qualified source reference/retained rows, chain, pinned liftOver/BEDTools and minimum-overlap declaration. Native mapping initializes compatible model parameters; it does not project the user's count matrix into human/mouse cCREs. |
| Reuse an accepted target model | Explicit model choice and compatible sparse CSR/CSC query, exact species/taxonomy, assembly and measurement semantics, complete model feature membership. | Exact checkpoint, vocabulary, training-corpus DF/cell count, configuration, native implementation and accepted binding. No new adaptation, SEAM or reference preparation is required. Same-plan model outputs can bind directly; historical reuse uses the existing trusted accepted-output path. |
| Embedding and downstream analysis | Eligible query/model and requested analysis. | Native target inference yields ordered 512-dimensional embeddings; existing neighbors, Leiden and UMAP tools compose through typed outputs. Current UMAP also accepts neighbors without a Leiden prerequisite, as established by [A.1](ua3.6.4a-umap-leiden-decoupling.md). |

`de_novo` still transfers pretrained shared model parameters and creates target
feature-specific parameters; it does not mean training the whole architecture
from scratch. Native-v2 supports fragment counts and declared binary accessibility,
with binary restricted to `de_novo`. Insertion counts are not qualified for
adaptation merely because neutral adoption can represent them. Historical v1
binary qualification remains its separate bounded profile; native-v2 defaults
do not reinterpret that historical identity or establish biological performance.

The public adaptation tool consumes one exact matrix. Internal joint-corpus
support is not a general multi-attachment user workflow. Canonical int64 CSR is
the current external-adoption representation contract, not a universal biological
requirement: broader lossless representations would need source/output conversion
provenance and owner verification. Target inference already accepts qualifying
integral floating-point sparse data. Neither path infers upstream measurement
history from values, filenames or species names.

## Genome identity, model identity and native ownership

`shared_genome.py` keeps `qualified-genome.v1` independent of H5AD/BED/peak/model
identity. Species/assembly, exact FASTA/FAI, contig geometry, provenance and
qualification pins define the resource. Explicit operator promotion requires
sharing provenance and reuse basis; uploads and user-trained models do not become
global defaults. Existing combined feature references remain feature-bound.

Selecting a shared genome for a different compatible peak collection creates a
private, differently identified ordered feature reference against the same genome
bytes. Native interval parsing and the existing reference owner validate exact
coordinates, without alias repair, dropping, sorting or curated-cCRE inference.
Plain FASTA supplementation generates a managed FAI when absent; a supplied FAI
is verified and preserved. Invalid FAI/content/contigs/bounds fail. Caller-supplied
provenance remains labelled as such; structure alone cannot prove assembly origin.

Genome reuse does **not** imply model reuse across different peak vocabularies.
`epizoo_target_inference.qualify_input` requires complete exact feature membership;
its lossless sparse column permutation into the model's recorded order is explicit
in `features_reordered` provenance, preserves rows/values and leaves the source
unchanged. Missing/extra/duplicate features, conflicting assembly or measurement
semantics fail. There is no feature fabrication, overlap projection or human/mouse
vocabulary restriction imposed on an otherwise qualified target model.

`native_posttraining.resolve` inherits the pinned `EpiZooXPostTrainConfig`:
500,000 attempts, learning rate `5e-5`, native save/log/retention settings, and
qualified loader batch 4. Explicit `max_steps`, `learning_rate` or batch 1–4 changes
only its corresponding field. Requested overrides and effective settings remain
recorded. CUDA/disk admission estimates necessary model/optimizer/storage bounds;
it is not an activation-memory, elapsed-time or convergence guarantee. No silent
batch/device/dtype/schedule reduction occurs. The default schedule was not run here.

`epizoo_adaptation/backend.py` calls native preprocessing, initialization and
`EpiZooXPostTrainer`. Its trainer subclass observes cancellation, finite losses
and applied/skipped AMP updates around `super().train_step`; it does not duplicate
optimization or loss algorithms. Exact vocabulary, training DF, corpus size and
token offset 4 feed native target inference. No scientific rewrite is warranted.

Historical profiles, receipts and authority remain exact. New training bundles
must qualify the current pinned adapter/native bytes; old bundles are not silently
renewed. Accepted-target inference has its existing narrower compatibility check,
while original adaptation recovery retains its full compatibility closure. Terminal
delivery consumes accepted evidence without training or structural reconstruction.
Interrupted training has no automatic optimizer/scheduler/RNG resume.

## Source binding and continuation finding

The historical live attempt in the Post-training record was re-inspected through
its preserved provider request/response JSON, without a provider call. Request 04's
catalog contains `input_path` but **no `input_sha256` and no reference-manifest
pair**. Adoption's source and reference ports expose no available complete request
source. The first candidate omits the source; the bounded repair explicitly selects
the incomplete `input_path` bundle. Its diagnostic is
`MISSING_REQUEST_SOURCE_MEMBER / select_complete_request_source_bundle`.

Thus the live request had an uploaded matrix and adaptation declarations, but
had not completed the typed reference handoff. The Interpreter selected execution
instead of reference supplementation; the Planner did not resolve that missing
prerequisite. This does not demonstrate a compiler rejecting a scientifically
complete source or a defect in the scientific owner. The historical harness also
landed in an existing Session, so this is not fresh-Session live qualification.

The current mechanism is concrete: `local_resources._h5ad_inputs` binds the selected
`input_path` and validates an offered digest against registration;
`resource_continuation._completed` supplies `input_sha256` internally from the
captured registered source when completing the reference prerequisite. It merges
only the selected compatible reference, preserving source, declarations and the
original objective. The ordinary HTTP test reaches real compiler/preflight through
this route and deliberately stops before training. Stale/source-substituted or
conflicting replies fail. Target-model InputSet continuation similarly preserves
the original H5AD; a selected target model is not redirected to a legacy default.

No deterministic language repair, automatic resource-to-role conversion or invented
reference/source completion is added. General first-turn source/model/reference
handoff and live selection remain application qualification work. Existing exact
bindings remain useful; this review does not assert that every sufficient arbitrary
attachment combination already reaches them.

## Findings and disposition

| Finding | Disposition |
| --- | --- |
| Genome sharing incorrectly tied to one peak collection | **KEEP existing correction**: separate shared genome plus private feature binding already handles distinct compatible collections. |
| Model-specific vocabulary, species/assembly, source checkpoint and measurement checks | **KEEP**: required correspondence and scientific eligibility, not removable genome restrictions. |
| Native defaults and fieldwise overrides | **KEEP existing correction**: resolved native configuration is already authoritative; no duplicate trainer to remove. |
| Historical live incomplete adoption source | **DEFER** live prerequisite/source handoff qualification; captured catalog lacked required bindings. Existing complete continuation passes. No proven local compiler repair. |
| Repeated genome scans | **DEFER** owner proof work if separately justified. Historical native execution recorded five reference builds/deep checks across owners. Shared-genome reuse avoids copies, not all scanning; bypassing these checks needs an explicit compatible owner proof, not deleting validation. |
| General noncanonical training H5AD, arbitrary attachments or automatic model/resource choice | **DEFER** explicit representation qualification and unified role/resource handoff. No silent conversion, first-match selection or model promotion. |
| Checkpoint size/delivery | **KEEP existing correction**: checkpoint-specific 5 GiB streaming already covers accepted full-vocabulary models. |

There is **no new SIMPLIFY implementation** in D. This report is its only new
repository file. The concrete earlier corrections remain intact rather than
being reopened or described as still missing.

## Outputs and verification

Exact accepted Session/Revision context controls checkpoint and safe metadata
delivery. `model_delivery.py` retains the **5 GiB** checkpoint limit, 8 MiB metadata
limit and existing smaller other-role limits. Streaming checks pinned bytes before
headers, uses bounded reads, reserves disk/concurrent transfers, rejects unsafe
paths and releases snapshots on completion/disconnect. Polling does not repeatedly
hash multi-gigabyte weights or load tensors.

Metadata records configuration, feature digest, training/source provenance and
explicit reuse requirements. **Checkpoint plus metadata is not a standalone
inference package**: exact `features.txt` and `document_frequency.npy` are not
included; corpus cell count, compatible native loader/runtime and eligible query
are also required. Embeddings/ordered cells, compact neighbors/analysis H5AD,
plots and safe reports remain distinct outputs with their own identities. Downloads
neither export Agent authority nor validate biological performance.

Eight existing focused modules pass **151 tests, 2 dependency deprecation warnings
in 54.64 seconds**, exit 0. `R/run_checks.py`, `command.json`, `focused.log` and
`focused-exit.json` record the run: agent-environment Python `-B`, `PYTHONPATH=src:.`,
no pytest/bytecode cache, isolated temporary/cache directories, all `RUN_*` gates
removed and no exclusions. Modules cover native configuration, public adaptation,
model delivery, accepted target binding, target inference qualification, Web
reference supplementation, target-resource continuation and new-target composition.

These are **small simulated/software and scripted-provider tests**. They exercise
actual small matrix/reference owners, FASTA/FAI qualification, native sequence
extraction, native configuration/meta-device capacity calculation, compiler/preflight,
publication/authority, HTTP transport and reopening. Training/model verification
backends are stubbed in public integration fixtures; target inference tests check
eligibility/provenance, not neural inference. Large-size cases test reservations
without creating multi-gigabyte files. No GPU training, SEAM, neural embedding,
live-provider call, native browser, download acquisition or full regression ran.

Historical evidence is reused with its original limits:

- [M14.4](m14.4-real-macaque-mapped-acceptance.md) and
  [M14.5](m14.5-real-zebrafish-de-novo-acceptance.md) establish bounded real mapped
  macaque and de-novo zebrafish execution and full-vocabulary checkpoint compatibility.
- UA3.6.4d's scripted native chain used an explicitly prepared real **16 × 17,922**
  matrix, applied **10 optimizer updates / 0 AMP skips**, then produced **16 × 512**
  embeddings, Leiden and **16 × 2** UMAP. Its presentation needed explicit recovery;
  ten updates establish technical execution, not convergence or generalization.
- The earlier workflow record includes an assisted successful live follow-up reusing
  an accepted model. This does not qualify new-model prerequisite selection. The
  later live Post-training attempt failed admission as described above.
- Final closeout supplies native Firefox downloads/restart with zero science;
  commit review supplies exact full streaming of the preserved 4,938,458,542-byte
  macaque and 4,018,926,698-byte zebrafish checkpoints. Neither transfer is repeated.

Final checks preserve all 676 tracked hashes, B/C report and operator hashes, all
1,274 eval metadata entries, all 200 protected historical file metadata entries
and 161 small-file hashes, and all 77 native tracked hashes. The three protected
model manifests retain the accepted M14.4, M14.5 and native Post-training digests
recorded in those reports. This is bounded preservation, not fresh full hashing of
every checkpoint. `preservation-final.json` records the comparison. The index stays
empty, repository refs remain unchanged, and whitespace checks pass. No scientific
identity, historical authority, EpiZoo source or deployment setting changes.

## Closeout and next phase

**D can close within this scope. A/B/C/D Scientific Integration Simplification is
complete as a bounded phase**, including A.1's accepted UMAP decoupling, B's retained
raw owners and documented context gap, C's clarification-text fix and D's preserved
adaptation paths. C's recorded batch setup error remains disclosed; D's passing
selection does not turn that earlier batch into a clean pass. No PARTIAL product
scenario is promoted.

The next unified input/application phase has concrete handoffs: B.1 source-bound
FASTQ/BAM context generation through the qualified owner; multiple files and exact
scientific roles; custom BED/raw routing into neutral owners; unique compatible
qualified-resource selection with explicit-choice preservation; natural missing-input
continuation; MAESTRO resources/groups; and EpiZoo source/model/reference bindings.
It also needs a scientifically qualified default Cell Selection policy preserving
explicit overrides and candidate-cell semantics, and understandable reports,
downloads and Session recovery. Broader training representations require explicit
owner qualification if included. These are not implemented here.

UA3.6.5 must then qualify actual ordinary-user, live-provider, real-data and browser
chains, including a correctly synchronized fresh-Session adaptation request. No
additional broad scientific audit or expensive training rerun is warranted by D.
