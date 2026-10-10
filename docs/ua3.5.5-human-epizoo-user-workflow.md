# UA3.5.5 — Human EpiZoo integration and first complete user task

The original human EpiZoo resource/default mapping is activated, and one genuine
32-cell human task completes through **native Firefox upload → live natural-language
planning → normal scientific execution → verified Run/Revision → browser evidence,
report and PNG**. The ten workflow boundaries below are **10 PASS, 0 FAIL,
0 BLOCKED**. Eight combined browser/runtime checks also pass. There is exactly
one model load and one foundation embedding invocation. No production source or
scientific algorithm changes; no commit or push occurs.

This qualifies one bounded user journey with an explicitly selected human
declaration. It does not establish utterance-only initial species binding, broad
language/biological reliability, or all 23 capabilities. UA3.5 is ready for scoped
closeout review on those stated terms; no automatic closeout or UA3.6 start occurs.

## A. Repository and operator state

On 2026-10-10, entry and final branch are `main`; HEAD, local `main` and local
`origin/main` remain `8bcad5b42cb812c89dfc232e3ceadea2fc5e44a7`. The clean index
is byte-identical. Entry contains 30 modified tracked files and 25 new
non-evaluation files. Final state retains those changes and adds only this report,
[the tests](../tests/application/test_human_epizoo_mapping.py) and
[the evaluation runner](../benchmarks/interactive/run_ua355.py): 30 modified tracked
files and 28 new non-evaluation files, plus evaluations. No prior UA3.5 delta is
overwritten or staged.

The actual [operator configuration](/home/likeyi/program/agent-web-config/operator-web.json)
gains exactly one `epizoo_resources` entry: `human-recorded-qualified`, species
`human`, default `true`, the existing checkpoint path and the human pins below.
Its qualification text cites original joint human–mouse pretraining, M14.2 source
qualification and bounded M12.2a canonical C0 inference. This new local catalog
handle connects reviewed resource bytes; it does not designate another model.
The original `mouse-recorded-qualified` entry remains exact. All other fields,
including both InputSets, model profiles, Groq default, credentials and upload
configuration, remain exact. File mode remains `0664`.

The configuration SHA changes from
`dd86afd663fc2345b542bd78fa4bf97a5ebbfaf6d2557e84736bcddb4a6d4389`
to `50ddd7846dd276f11465f98040b285218fb459acbe480bb8aa90350f2283e880`.
A restorable `copy2` backup resides at
`/tmp/agent-ua355-preservation-oo4_1axk/operator-backup/operator-web.json`, inside
two `0700` private directories. The full configuration is never copied into Git.
[Change evidence](../evals/ua3.5.5_human_user_workflow_2026-10-10/diagnostics/operator-change.json)
records the exact append and preservation checks.

The unchanged deployment has only its Groq profile and mouse H5AD companion.
The isolated acceptance server additionally admits the existing exact F1 profile
and a human H5AD companion containing only `species=human`, through ordinary
constructor/InputSet interfaces. Firefox actually selects both controls and leaves
the EpiZoo resource selector at **Use applicable configured default**. These are
evaluation-local choices, not hidden defaults or permanent operator edits. Device
is omitted and the existing owner supplies `cuda:0`.

## B. Shared model, human mapping and invocation ownership

| Original resource | Established and consumed SHA-256 |
| --- | --- |
| Joint `pretrained_EpiZoo.pth` | `6b2d13fdbd54a9b0d56efa5afa81bc4832b813f4eac8662e4d93cc08c9d9b39a` |
| Human `cCRE_frequencies_human.npy` | `8b576fa4fc60a1e2fc77607ffacff2883441def3d8cb8225776b71ad6c00e80b` |
| Human `cCRE_filter_idx_human.csv` | `994d9c3e87208074e695c4c418b28d9587dd8991ad033cf33e62f96ceebc7875` |

M14.2's retained original source bundle, identity
`9fa015094881ca168ea281c8234f374e3f704c90a998beb09ec4374404d7982a`, qualifies
both original vocabularies and the same joint checkpoint. The
[M12 closeout](m12-matrix-level-input-closeout.md) supplies historical real human
canonical C0 inference, independently of adaptation. Current small auxiliary
hashes match those records. Registration reuses the established large-checkpoint
identity without an extra model load or preservation rehash. The actual owner
subsequently checks the bytes it consumes and publishes matching expected/actual
identities. No retraining, adaptation or redundant model qualification occurs.

`select_epizoo_resource` now finds the unique declared-human default. Explicit
qualified choices retain precedence; absent/ambiguous defaults and species/pin
conflicts still fail closed. Human and mouse use one shared owner and checkpoint,
with their respective original auxiliaries. Arbitrary human peak matrices remain
outside the full-axis contract.

The invocation path is Registry `epizoo_embed_cells` →
[`analysis.epizoo_embedding`](../src/agent/tools/analysis/epizoo_embedding.py) →
[`epizoo_cache`](../src/agent/tools/models/epizoo_cache.py) and
[`models.epizoo`](../src/agent/tools/models/epizoo.py) → original
`epizoo.data.processing`, `epizoo.data.datasets`, `epizoo.models.epizoo` and
`epizoo.inference.embeddings`. Those modules resolve to
`/home/likeyi/program/EpiZoo`, clean at
`029cd631d0f9806a646c4a3b42ce10b958f2b67f`.
Agent directly reuses original `EpiZoo`/`EpiZooConfig`, TF-IDF, filtering, sentence
generation, dataset sampling/collation and embedding extraction. There is no
Agent copy of the transformer, normalization formula, token ranking/truncation
algorithm or model-forward loop. Agent's adapter owns invocation, strict compatible
loading, sparse/input/resource checks, cache lifecycle, reproducibility, artifacts
and provenance. The original preprocessing order and human constants remain
those in the [interface audit](epizoo_interface.md).

There is existing downstream **composition overlap**: the independently accepted
M6.1 neighbors/Leiden/UMAP ports call Scanpy directly and overlap the original
EpiZoo convenience wrapper's AnnData/neighbors/UMAP composition. They do not
reimplement Scanpy numerics. The original helper recomputes neighbors and sets a
different global seed; its clustering helper searches label-optimized Louvain.
They are not interchangeable with the accepted separate fixed-Leiden contracts.
This actual overlap is recorded without a refactor. All owners remain unchanged.
[Read-only invocation review](../evals/ua3.5.5_human_user_workflow_2026-10-10/diagnostics/epizoo-invocation-review.json)
identifies exact source lines, resolved modules and prior acceptance.

## C. Genuine input and observed user workflow

The retained original M12 C0 input still exists. Its exact bytes are copied into
the private input workspace, not regenerated or projected: 80,541,640 bytes,
SHA-256 `0c601bf017053f82fa34273156f647148c3ccddecf6646fcd9fcb4cc4aabe264`.
It is a **32 × 1,355,445 int64 CSR**, 336,129 nonzeros, with the complete original
human feature axis, saved M12 cell selection/order and no fabricated labels.
An independent preparation-only comparison checks every sparse row value/index,
every feature name/order and every cell identity against the accepted 1,119-cell
human canonical source. It invokes no scientific owner or raw reconstruction.
The input is a bounded representation, not a new matrix authority publication.
[Bounded trust record](../evals/ua3.5.5_human_user_workflow_2026-10-10/diagnostics/input-trust-projection.json)
retains source pins and order/conservation digests; full cell records remain private.

Firefox submits exactly:

> Analyze these human scATAC-seq cells with EpiZoo, construct their neighbor graph,
> perform Leiden clustering, and compute UMAP.

| Boundary | Observed result | Status |
| --- | --- | --- |
| 1. Genuine compatible input | Exact retained canonical human input; normal owner admits sparse counts, full dimension and retained-name order | **PASS** |
| 2. Native upload/registration | Actual file input and Upload button; registered source has the exact input digest; upload creates no models/science/Revisions | **PASS** |
| 3. Natural-language submission | Actual composer sends one ordinary request with selected upload, F1 profile and explicit human declaration | **PASS** |
| 4. Live semantic decisions | Interpreter `execute_plan/compute_cell_umap`; capability `embedding_analysis`; real four-step candidate; 13 output selections | **PASS** |
| 5. Human resources/defaults | Unique human catalog default; original checkpoint/human auxiliaries; optional numeric inputs omitted | **PASS** |
| 6. Compilation/preflight | Exact same-plan references compile; Planner and durable normal executor preflight pass | **PASS** |
| 7. Scientific dispatch | Unchanged ordinary `PlanExecutor`; four registered owners called exactly once; no withholding executor | **PASS** |
| 8. Scientific owner outcomes | Genuine human embeddings, neighbors, Leiden and UMAP all succeed with normal validation | **PASS** |
| 9. Verification/publication | Every step, Run verification and persisted preflight pass; one active Revision, generation 1 | **PASS** |
| 10. User result access | Accepted details, all 13 evidence selections, safe report projection and PNG are opened in Firefox; history/Revision survive refresh/server restoration | **PASS** |

The selector chooses 13 of 56 offered step/key pairs with globally unique names.
Persisted `SessionTurn.selections` and Revision outputs exactly match the live
response. No names/keys are repaired, no plan is automatically completed, and
there is no planning-approval turn. The human declaration comes from the selected
typed companion; this does not prove initial utterance-only species extraction.

[Live results](../evals/ua3.5.5_human_user_workflow_2026-10-10/live-complete/results.json),
[Run](../evals/ua3.5.5_human_user_workflow_2026-10-10/live-complete/workflow-run.json)
and [Session](../evals/ua3.5.5_human_user_workflow_2026-10-10/live-complete/workflow-session.json)
retain raw choices separately from their ordinary admission and verification.

## D. Actual scientific results and provenance

| Owner | Actual verified result | Effective existing settings |
| --- | --- | --- |
| EpiZoo | Finite float32 **32 × 512** embeddings; exact ordered ID sidecar | Human original preprocessing; batch 4, maximum length 8,192, random sampling, seed 0, workers 0, float32, CUDA/AMP, overwrite false |
| Neighbors | 32-cell compact H5AD; **542 connectivity** and **448 distance** nonzeros; 512-dimensional representation | k=15, Euclidean, seed 0, transformer `none`; no raw-feature densification |
| Leiden | **4 clusters**, key `leiden`; ordered compact H5AD | Weighted fixed Leiden, resolution **1.0**, seed 0 |
| UMAP | Finite float32 **32 × 2** coordinates, key `X_umap`; original cells/order preserved | 2 dimensions, min_dist **0.5**, spread **1.0**, seed 0 |

No optional numeric argument appears in the compiled/resolved downstream inputs.
Owner results and compact-artifact provenance record the effective values. The
existing reviewed client evidence projection explicitly reports
`clustering_resolution_origin=existing_owner_default`; that origin is not invented
in the base evidence file. EpiZoo's `epizoo-resource-provenance.v1` records
`human-recorded-qualified`, exact matching expected/consumed hashes above and all
fixed inference settings. These are software/scientific-contract results on the
bounded control, not cell-type, clustering-quality or biological generalization claims.

Accepted identities are:

- Session: `8d355d7d-b116-4724-9b6e-4e2520f78f6e`
- Turn: `5f9613be-0340-4b89-a21f-12e05b6f9d02`
- Run: `turn-bb203fea9c143a8c640e58e8a400578d3ca31d044380c6f4354431de2cc6e912:run`
- Revision: `1f6ce7fbbdbc60c3fbf9bbccdb4fc6f8dfbd5068310a378fd6d84d9bf9dac9d0`

Run lifecycle is `SUCCEEDED`, facade status `succeeded`, turn `activated`, errors
empty and generation 1. The completion pins bind accepted evidence
`bd300e6dc00a457ef262bb35c5e15e2cc1a32b9890e637a0f8e99cd44404cb6b`, original report
`736b173277c8ff6642e15759c4dc8c2fce8efc77a02757575d677121eec3d434` and UMAP PNG
`ab9b53217367e453cadeee473395c229214c3407287362670cedb2f1dc494777`.
The Web report is a labeled safe projection, not raw original Markdown.

Runtime checks find CUDA available on the RTX 4090 with approximately 23.3 GiB
free immediately before dispatch. Peak **Torch allocated** memory is 11,737,160,704
bytes, **10.93 GiB**; this is not a whole-device peak measurement. One backend
load, one resource load and one `embed_cells` invocation are observed. Each
registered scientific step has attempt count 1. Normal execution/completion
materialization invokes `verify_step` 60 times and `verify_run` 15 times; verification
is not claimed to occur only once. All observer counts remain unchanged across
refresh and a new server/application instance in the same process.

## E. Staged validation, live bounds and preparation diagnosis

Stage A passes **276 tests, three existing dependency warnings in 57.50 seconds**,
exit 0. It covers human/mouse catalog selection, explicit precedence, missing and
ambiguous resources, mismatches, changed pins, H5AD dimension/order, species
continuation, owner defaults and output selection. The exact 16-target command,
environment and log are retained in
[Stage A command](../evals/ua3.5.5_human_user_workflow_2026-10-10/diagnostics/stage-a-command.json)
and [log](../evals/ua3.5.5_human_user_workflow_2026-10-10/diagnostics/stage-a.log).

Stage B passes **2,050 tests, nine skipped, three existing warnings in 558.45
seconds**, exit 0, without exclusions:

```bash
PYTHONPATH=src:.:/tmp/agent-ua355-stage-b-p1y6hck7/observer PYTHONDONTWRITEBYTECODE=1 UA354C_ADMISSION_EXCEPTION_LOG=/tmp/agent-ua355-stage-b-p1y6hck7/admission-exceptions.jsonl /home/likeyi/anaconda3/envs/agent/bin/python -B -m pytest tests/application tests/web -q -p no:cacheprovider --basetemp /tmp/agent-ua355-stage-b-p1y6hck7/pytest-tmp --tb=short -p ua354c_admission_observer
```

Both stages disable all opt-in `RUN_*` gates, make zero provider calls and preserve
all 462 frozen source/test/static/resource files during each run. The existing
BAM exception observer delegates once and rethrows unchanged; it records zero
targeted exceptions. UA4c's earlier uncaptured BAM failure remains unexplained;
this pass establishes nonrecurrence, not a root cause or unrelated repair.

The new scripted tests separately qualify species pending/resumption, original
task/source/device and exact human/mouse resource admission while deliberately
withholding science on a structural tiny H5AD. They are not positive foundation
inference. An additional exact `Analyze this dataset using EpiZoo.` → `These are
human cells.` single-embedding witness is recorded after the full regression,
without a second genuine inference. The final supplemental command is:

```bash
PYTHONPATH=src:. conda run --no-capture-output -n agent python -m pytest -q tests/application/test_human_epizoo_mapping.py
```

It passes **5 tests in 1.85 seconds**, exit 0, including the four original cases
and one added exact-utterance case. The full Stage B result precedes that test-only
addition; no broader rerun is claimed. An initial supplemental development run
fails only an unnecessary assumed seven-call assertion; the scripted fixture makes
eight calls. Removing that count assumption changes no admission assertion or
production behavior. Both [final](../evals/ua3.5.5_human_user_workflow_2026-10-10/diagnostics/exact-species-command.json)
and initial commands/logs remain in diagnostics.

The genuine workflow makes **4 model-interface calls and 4 provider completions**:
one Interpreter, one capability selection, one detailed Planner and one output
selection. All identify `openai/gpt-5.6-sol`, one upstream attempt and an unmodified
pipeline. Usage is **8,070 prompt / 1,309 completion tokens**; provider-reported
cost **$0.032751**, conservative reservation **$0.266264** against the fixed $5
ceiling. One exact catalog lookup confirms availability and accepted public prices;
its evidence is reused after the preparation failure below, with no second lookup.
SDK retries, semantic repairs, failovers, provider failures and guard blocks are zero.
Inherited limits remain 48 total completions, six per turn, 4,096 output tokens,
150,000 conservative input byte/token units, strict parameters, exact routing and
one dispatch per turn/schema. No balance claim is made.

The actual successful command, exit 0 in 113.59 seconds, is:

```bash
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 /home/likeyi/anaconda3/bin/conda run --no-capture-output -n agent python -B -m benchmarks.interactive.run_ua355 --live --output evals/ua3.5.5_human_user_workflow_2026-10-10/live-complete --workspace /tmp/agent-ua355-live-20261010-complete --source-h5ad /tmp/agent-ua355-human-input-kk2y0q98/human_pbmc_m12_canonical_32.h5ad --operator-config /home/likeyi/program/agent-web-config/operator-web.json --discovery-json evals/ua3.5.5_human_user_workflow_2026-10-10/diagnostics/exact-discovery-reused.json
```

The [command record](../evals/ua3.5.5_human_user_workflow_2026-10-10/diagnostics/live-complete-command.json)
also records isolated NumPy/Matplotlib cache directories. For reproduction use
fresh output/workspace paths; omit `--discovery-json` to perform a new exact lookup.
The native helper source, raw responses, logs and screenshots are archived.
The runner uses the ordinary executor; its Registry callbacks only count and
delegate original functions, with an extra one-foundation-entry ceiling. No
semantic decision, prompt, scientific owner or result is patched.

The [preparatory attempt](../evals/ua3.5.5_human_user_workflow_2026-10-10/live/results.json)
successfully uploads but makes **zero model constructions, interface calls,
completions or scientific calls**. Its evaluation-only Marionette `fetch` observer
uses an invalid Window receiver, so POST never reaches the server. Two read-only
normal admission checks pass without state changes. An isolated static echo
reproduces `TypeError: 'fetch' called on an object that does not implement interface
Window.`; unmodified fetch succeeds. Removing that observer is the only corrective
behavior change, confined to the new benchmark. Exact original runner bytes, timeout
snapshot/log, independent reviews and
[reproduction](../evals/ua3.5.5_human_user_workflow_2026-10-10/diagnostics/fetch-observer-reproduction/results.json)
remain intact. This is a resolved harness defect, not a semantic retry, production
admission failure or repeated foundation run.

## F. Preservation

All **589 original repository files**, **1,002 prior evaluation files**, 198
inventoried scientific outputs and 208 inventoried real-state files retain SHA-256,
size, mtime and mode. This includes every earlier UA3.5 change and scientific owner.
The only operator difference is the declared human catalog append; its backup is
restorable, permissions and private fields preserved. Source human matrix/M12
input and auxiliary resources retain their pinned bytes. The original checkpoint's
5,231,645,507-byte size, mtime, mode and consumed identity remain unchanged.
Configuration work creates no alternate checkpoint. During normal loading the
unchanged owner makes and removes its existing private immutable consumption
snapshot; this required identity safeguard is not bypassed.

New inputs/scientific outputs and Session state remain only in private disposable
workspaces outside Git, principally `/tmp/agent-ua355-live-20261010-complete`.
No H5AD, embeddings, cell-ID sidecar or checkpoint is added to Git. Credentials
are read from the existing Conda environment, never printed or copied; SDK headers
and client representations are not saved. New text artifacts are screened against
configured private values. The original EpiZoo repository remains clean at its
same revision. No reset, clean, stage, fetch, commit or push occurs.

The [preservation audit](../evals/ua3.5.5_human_user_workflow_2026-10-10/diagnostics/final-preservation-audit.json)
and [independent saved-record review](../evals/ua3.5.5_human_user_workflow_2026-10-10/diagnostics/independent-post-live-review.json)
retain exact evidence; the latter passes all 21 checks without replay. Native refresh/server
restoration preserve conversation, active Revision/generation, all workspace file
digests and semantic/scientific/verification counters. This is a new server and
application instance within one process, not a claimed cold Python restart.

## G. Remaining limits

The tested human file must satisfy the existing full-axis sparse count-like
contract. A human declaration or checkpoint identity cannot make arbitrary peak
H5ADs compatible. This slice includes no annotation, ground truth, biological
interpretation or calibrated clustering/cell-calling policy.

The one-turn live case supplies species through a selected reviewed declaration
control. Initial `ExecutePlan` does not itself carry a species declaration solely
from the word “human”; without typed species the existing pending/answer path
remains necessary. That exact human continuation is tested deterministically,
not through a second live/scientific run. The unchanged deployment receives the
human mapping but not the evaluation-only F1 profile or human companion. Its
ordinary missing-species continuation can now resolve the admitted human tuple.

User-visible accepted summaries, safe report and PNG are demonstrated. Generic
raw embedding/ID/compact-H5AD download and cross-turn `PriorOutputRef` authority
for these processed outputs remain unsupported. Client evidence explicitly says
`accepted_persisted_summary` and `Authority scope: Unavailable`; presentation is
not fresh scientific authority. No transport allowlist or authority adapter is
expanded. The existing result button/Revision navigation makes this exact task
accessible without those additional contracts.

## H. UA3.5 recommendation

The human resource mapping is **complete**. The bounded real scientific task is
**complete**. The combined native-browser + real-provider + real-science journey
is **genuinely verified** under the explicit declaration and profile choices
above, rather than assembled from separate witnesses. No narrow user-facing
blocker remains for this qualified slice.

Recommend scoped UA3.5 closeout review of the accepted shared mechanisms and this
complete task, retaining explicit scientific declarations, qualified resources
and the documented authority/export boundaries. Broader initial declaration
collection, datasets/capabilities, multi-turn/backend-language reliability and
biological interpretation remain separately scoped UA3.6 work. This single
combined success does not close that broader acceptance automatically.
