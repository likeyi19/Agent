# USI-6 — EpiZoo standard analysis and cross-Revision scientific comparison

2026-10-16 (Asia/Shanghai). Scientific/application acceptance uses scripted
language decisions and real EpiZoo/Scanpy execution. It does not qualify live
provider interpretation or comprehensive native-browser behavior. Final
regression and preservation disposition are recorded below.

## A. Baseline and preservation

HEAD, branch `main`, local `origin/main`, and the independently queried remote
`refs/heads/main` were all
`89385ffaffcb3d24ff17a0a0f5d367d07b570fa2`,
`Complete USI-5 EpiZoo cross-species and post-training handoff`.
The tracked tree and index were clean. Initial status was solely `?? evals/`.
No staging, commit or push is authorized or performed. Production configuration
is unchanged and no resource download was needed or performed.

Evidence root **R** is `/home/likeyi/agent-usi6-07lldlo1`; the final scientific
workspace and drivers are under **F** = `R/final`. The preliminary execution at
`R/workspace` is preserved separately. Identical Session/turn names in these two
isolated stores can produce identical Revision IDs; the workspace, accepted
Run/step digests and artifact pins distinguish the actual evidence. Results from
the two stores must not be conflated.

`R/baseline.json` captures all 708 tracked hashes and 63,908 protected file
entries inherited from the USI-5 preservation scope plus USI-5's evidence tree,
`evals/` and operator configuration. Size, mtime and mode cover every recorded
file; files at most 2 MiB also have hashes. This is a bounded audit, not a full
rehash of every historical dataset. Consumed human input, original checkpoint
and auxiliaries have separate exact known-hash checks. Approximately 14 TiB free
disk was available. No historical scientific output is rewritten or qualified
retroactively.

## B. Existing capabilities reused

The audit covered README/AGENTS, USI-1–5, the UA3.6.4a real workflow, scientific
integration simplification, UMAP decoupling and dialogue closeout, relevant
UA3.6.4d records, M15 Session/prior-output contracts and M16 evidence/comparison.

| Boundary | Existing behavior retained |
| --- | --- |
| Upload and roles | Session-scoped streaming H5AD upload, exact registration, typed source handles and immutable submission capture. |
| Scientific resources | Original qualified EpiZoo catalog and USI-5 target-model/genome mechanisms. Standard human inference requires no FASTA/FAI, adaptation, mapping or manual InputSet. |
| Parameters | Registry scalar declarations, exact utterance grounding, explicit-value precedence and owner defaults. No natural-language parameter parser was added. |
| Execution | Existing scoped Planner, semantic compiler, preflight, sequential runtime and 23-tool Registry. |
| Scientific owners | Original EpiZoo inference and Scanpy neighbors, Leiden and UMAP. UMAP already accepts a neighbor graph without Leiden. |
| Historical reuse | Exact accepted neighbor publication through existing schema-2 authority and PriorOutputRef. Readable embeddings/clustering/UMAP do not acquire executable authority. |
| Evidence and answers | Existing accepted Run/step/evidence checks, bounded facts, exact output groups, claim/pair rendering and post-generation freshness checks. |
| Delivery and recovery | Existing scientific-artifact handles, private pinned snapshots, safe H5AD projection, reports/PNG, generation-checked navigation and durable Session reads. |

The standard tools compose supported subsets through the Registry. A raw H5AD
containing an arbitrary `obsm` array is not an already-supported embedding-adoption
interface: `build_cell_neighbors` consumes qualified ordered NPY/ID inputs or
their accepted producer binding. This work adds no implicit extraction or
qualification of arbitrary uploaded embeddings.

## C. Reproduced gaps

1. `R/gaps.log` records four failing tests before changes. Parameter pairs for
   independently accepted neighbors, Leiden and UMAP were withheld unless direct
   measurement compatibility was proven. Descriptive settings require a weaker
   claim than aligned scientific outputs. An explicitly named older Revision
   outside immediate navigation/predecessor scope was also absent from the offered
   inventory.
2. The preliminary real attachment-group run succeeded scientifically but its
   embedding arguments lacked `expected_resource_identity` and its result lacked
   `resource_provenance`. The original-model catalog resolver was applied only to
   singleton registered H5AD composition. The explicit group-role path bypassed it
   and used native defaults. `R/run-initial.json` preserves that diagnosis.
3. The same real run produced independent Leiden and UMAP files, but HTTP offered
   only neighbors and UMAP. Embedding-only and Leiden-only accepted results lacked
   pinned compact downloads. Requiring Leiden to feed UMAP merely to download its
   labels would reintroduce the scientifically unnecessary dependency.

The preliminary run and its passing comparisons are not substituted for the final
catalog-pinned and fully downloadable acceptance.

## D. Implementation and architectural simplification

| Files (under `src/agent` unless stated) | Change and ownership |
| --- | --- |
| `application/scientific_dialogue.py` | Offer bounded older Session revisions through existing result handles; distinguish reviewed descriptive method/settings/dimension pairs from qualified direct operands; add UMAP scope explanation. |
| `application/turn_decisions.py` | Clarify that scientific result handles can identify older versions/branches and that an incomplete bounded history inventory must not cause target substitution. |
| `application/dialogue_execution.py` | Apply the existing original-model resolver to explicit embedding attachment roles with declared species and a configured catalog. Replay its existing captured catalog receipt on reads. |
| `application/local_resources.py` | Require exact species/checkpoint/resource-pin consumption for a group with a selected original EpiZoo resource, before execution and at acceptance. |
| `tools/analysis/epizoo_embedding.py` | Publish a compact H5AD containing the existing embedding values and ordered cells, using existing atomic H5AD publication. No inference or preprocessing changes. |
| `tools/analysis/embedding_analysis.py` | Pin newly produced Leiden H5AD bytes, as UMAP already does. No clustering or layout algorithm changes. |
| `orchestration/verifier.py` | Share the existing payload-digest check and independently check embedding H5AD conservation against NPY/ID outputs. |
| `application/analysis_delivery.py` | Extend the existing safe compact export to embedding and clustering stages; preserve the legacy UMAP variant when stage is absent. |
| `application/interactive.py`, `interactive_schemas.py`, `web/static/app.js` | Offer the existing accepted steps through the same HTTP delivery mechanism and admit the two exact new filename/label descriptors. No layout redesign. |
| `tests/application/test_standard_analysis_comparison.py` | Descriptive compatibility, older history, bounded inventory, corruption, missing operands and unavailable metrics. |
| `tests/web/test_standard_analysis_handoff.py` | Attachment catalog pins, explicit zero, independent UMAP, four downloads, resource substitution, tamper rejection and absence of new executable authority. |
| `tests/unit/orchestration/test_verifier.py` | Public verifier rejects either incomplete compact-embedding export field pair with a structured failure, while the complete pair passes. |

The optional captured `historical_revisions` list contains existing Revision IDs
only. It permits deterministic replay of new bounded reference inventories while
old captures keep their original meaning. Whole revisions are added within the
existing 128-output limit; required current/parent/previous/predecessor scopes are
not dropped. `history_complete` reports omission to the interpreter. Inventory
order grants no scientific preference, and the LLM still selects an offered handle.

`descriptive_accepted_operands` is an ephemeral mode of the existing pair context,
not another comparison service or persisted scientific schema. Its reviewed
fields describe methods, settings and dimensions; population/cluster counts do
not become direct pairs without the existing qualification. Tool/result/recovery
contracts must match. Equal cluster IDs and coordinate arrays are never paired.

New embedding results add optional `analysis_path` and `analysis_sha256`;
new Leiden results add optional `analysis_sha256`. These are necessary file pins,
not schema-2 authority. The delivery manifest adds a closed stage discriminator
only for the new variants; absence retains the old UMAP contract. Historical
unpinned outputs stay readable but are not retrospectively made downloadable.
The existing compact export strips paths and arbitrary source metadata, preserves
cells/values/real labels and uses the existing size, snapshot and digest checks.

No new tool, model catalog, Session/Revision model, authority format, scientific
comparison algorithm, metric store or execution engine is introduced.
`src/agent/tools/models/epizoo.py` and native EpiZoo are unchanged.

## E. Real standard-analysis acceptance

`F/real_workflow.py` uses actual HTTP upload/turn/poll/download endpoints and the
ordinary application, with scripted interpreter, capability, plan, output and
answer selections. All scientific computations and verifiers are real.

Input is the preserved UA3.5.5/UA3.6.4a human PBMC canonical C0 subset:
`/tmp/agent-ua355-human-input-kk2y0q98/human_pbmc_m12_canonical_32.h5ad`.
It contains 32 cells, 1,355,445 ordered canonical features, int64 CSR and 336,129
nonzeros, with accepted fragment-record-count provenance. Size is 80,541,640 bytes;
SHA-256 is `0c601bf017053f82fa34273156f647148c3ccddecf6646fcd9fcb4cc4aabe264`.
This is a legitimate bounded historical subset, not simulated biological data.

The ordinary user supplies the H5AD and an explicit human declaration. The
configured original resource `human-recorded-qualified` resolves automatically.
Final accepted expected and actual pins agree:

| Component | SHA-256 |
| --- | --- |
| Original joint checkpoint | `6b2d13fdbd54a9b0d56efa5afa81bc4832b813f4eac8662e4d93cc08c9d9b39a` |
| Human frequencies | `8b576fa4fc60a1e2fc77607ffacff2883441def3d8cb8225776b71ad6c00e80b` |
| Human retained-feature indices | `994d9c3e87208074e695c4c418b28d9587dd8991ad033cf33e62f96ceebc7875` |

Inference retains batch 4, maximum length 8192, seed 0, random sampling, zero
workers, CUDA:0, float32 parameters and CUDA AMP. Peak Torch allocated memory is
11,737,160,704 bytes (10.93 GiB), not a whole-device peak measurement. No complete
scATAC matrix is densified; no training, SEAM or mapping is performed.

| Turn | Actual computation and accepted outcome |
| --- | --- |
| Initial | EpiZoo once → neighbors once; independent Leiden and UMAP from the graph. Finite 32 × 512 embeddings; k=15 Euclidean; explicit resolution 0.5 gives 3 clusters; UMAP min_dist=0.5, spread=1, seed=0, 32 × 2. |
| Revised | Explicit resolution 1.0, unchanged graph and other settings; only Leiden and independently requested UMAP execute. Four clusters, unchanged UMAP settings. |
| Zero | Only UMAP executes from the retained graph with explicit min_dist=0.0. |

Initial downloadable files are `embeddings.h5ad`, `neighbors.h5ad`,
`clusters.h5ad` and `analysis.h5ad`, plus the existing report and PNG delivery.
Revised/zero results expose their produced files and explicitly retained graph.
Embedding and Leiden are downloadable independently of UMAP. The exported
UMAP in this witness contains no invented Leiden labels. `F/downloads.json`
records every actual handle, filename, size and digest; `F/run-*.json` records
resolved arguments, accepted output pins, verification and lineage.

These are SHA-256 values of the actual safe HTTP downloads, distinct from their
source-artifact pins in `downloads.json`:

| Download | SHA-256 |
| --- | --- |
| Initial embeddings | `9af327d5fbdcfa980dd08716991a8b8ecb66a3a6090db9337434b728d502edf8` |
| Retained neighbors | `7c4cbd868c529429be61098a920e78f4c5a60b2caad2d2798ade03f6f0d36879` |
| Initial Leiden | `14dd2f95d96beb5beee8b9a72266da763b8d35ea0b690651d481eba4a6c9c3d8` |
| Initial/revised UMAP | `15151344ce56920d3325d5ed8756e6c6ee24f1ca50deb5e7b544c263dc8d1593` |
| Revised Leiden | `0644acca6dc3ac3304b4ea77966c1ff97e1ceb1f19cf8433c704c6318af948cd` |
| Zero-min_dist UMAP | `91ee47c5d3ff5dd58e3b484997876710adbd88ac1d021493fe2a1959c70ca87b` |
| Branch Leiden | `d5d8d6506391a010844ec87b4700fec31cc085b0b5b97a9bf38c558c122c2fa9` |

## F. Cross-Revision comparison acceptance

Within `F/workspace`, Session `test` has the following initial three revisions:

| Version | Revision ID |
| --- | --- |
| 1, initial | `3b279a9a9cdd5b5b0c5b734fec7edb819b0e159a907460a44cb8049d50241b72` |
| 2, revised | `296fc5617057b8d21cf9673ed2f045baf9d883ffb4bd48719c30cf66e59e8bd6` |
| 3, min_dist zero | `e56e9343cff7b21bd58b44e504d9b32dcb6b291c94287b45fc0c167d8c411791` |

`F/comparison.json` contains an accepted answer binding the distinct clustering
outputs of versions 2 and 1: 32 cells in each, resolutions 1.0 versus 0.5, and
4 versus 3 clusters. The existing exact qualified-neighbor identity establishes
the common graph population. Claims retain exact Revision, output locator,
accepted step, evidence digest and source pointer. The answer explicitly rules
out cluster correspondence, cell-type identity, significance and superiority.
No cluster membership overlap or label alignment is calculated.

`F/umap-answer.json` binds the UMAP outputs of the same two revisions, with
equal min_dist=0.5, spread=1, random seed=0 and dimensions=2. The answer explains
that independently generated layouts do not establish aligned biological
coordinates; rotation, reflection or scale alone is not biological change.
The later zero-valued request is a new computation, not work hidden in an answer.

Before and after each answer, instrumented EpiZoo, neighbors, graph verifier,
Leiden and UMAP counts are unchanged. The complete initial/revised/zero sequence
totals one inference, one neighbor production, one independent neighbor
reconstruction, two Leiden and three UMAP calls. This is not a claim that every
structural verifier or presentation check was called only once.

## G. Multi-turn history, branch and recovery

`F/inspect_recovery.py` independently reopens all H5AD outputs and downloads,
checks exact source cell order and embedding values, verifies actual labels and
finite coordinates, navigates to version 1 and returns to version 3. A stale
generation navigation receives HTTP 409. The first three complete Revisions and
32 nonempty scientific/RunStore files remain unchanged; only ordinary navigation
records advance Session generation from 3 to 5.

`F/branch.py` then navigates to version 1 and creates a sibling branch by running
only Leiden at resolution 0.8 on the existing graph. It compares that fourth
revision with explicitly named version 2, which is not its parent. The new bounded
history inventory offers the exact results; there is no navigation-based target
substitution. The branch runs Leiden once, with zero inference, neighbor
production/reconstruction or UMAP calls; its comparison performs zero science.
Original revisions are unchanged. `F/branch-audit.json`, `branch-run.json`,
`branch-comparison.json` and `branch-downloads.json` retain exact identities.
Version 4 is `b52085d4f434e0a53414603f0e624622cabde445913edcff7e290b262c991c93`;
its parent is version 1. Its accepted comparison reports 32 versus 32 cells,
resolution 0.8 versus 1.0, and 3 versus 4 clusters, on the same qualified graph.

Fresh-process recovery uses `--reopen`, with providers and instrumented scientific
entry points forbidden. Historical evidence and downloads retain exact bytes.
Both the three-revision recovery (`F/reopen.log`) and later four-revision branch
recovery (`F/branch-reopen.log`) pass, with zero guarded calls.
Completed recovery is qualified; interrupted model-training recovery and a new
scientific crash/recovery protocol are not claimed.

## H. Scientific and product limits

- Descriptive pairs do not establish a shared cell population, compatible model
  coordinates or matching biological groups. Direct Leiden count comparison
  retains the existing exact qualified-graph boundary.
- Published compact evidence has no cell-level contingency/overlap, silhouette,
  significance, calibrated cluster quality or independent cell-type validation.
  These facts are unavailable; no new metric is computed to fill the gap.
- Full source/model hashes remain available in accepted evidence and lineage.
  The answer context does not expose private paths or invent missing assembly,
  measurement-history or model-equivalence facts. A changed model is not made
  equivalent merely by equal dimensions or species.
- Current embedding/Leiden/UMAP evidence and download pins are not reusable
  schema-2 execution authority. Valid neighbor reuse is demonstrated; no authority
  sidecar is fabricated for other outputs.
- Older history is bounded to 128 offered outputs, with explicit incomplete
  coverage. Ambiguous or unavailable references require clarification/navigation.
- Missing/incompatible choices in the configured original-model catalog fail
  without substituting another checkpoint. Existing typed prerequisite/resource mechanisms remain;
  this work does not add new resource acquisition or arbitrary embedding adoption.
- Language decisions are scripted. Live-provider quality, browser-wide acceptance
  and broader UI design remain UA3.6.5/USI-7 work. No provider call or retry is made.

## I. Testing and preservation

All tests use the existing agent environment, isolated basetemps, no pytest cache
in the repository, and disabled `RUN_*` gates unless separately stated. No full
historical eval collection is executed. Counts below overlap and are not summed.

| Invocation | Outcome |
| --- | --- |
| Initial gap reproduction | 4 failures before the dialogue correction (`R/gaps.log`). |
| Initial dialogue regression | 97 passed in 63.18 s. |
| Expanded comparison negatives | 10 passed in 9.65 s. |
| Handoff/export/owner tests | 120 passed in 21.19 s, three dependency warnings. |
| Delivery/verifier regression | 96 passed in 177.89 s, three dependency warnings. |
| Final focused selection | 220 passed, 3 baseline failures in 187.23 s; all frozen source/test files unchanged. |
| Exact-HEAD baseline reproduction | All 12 previously documented failure nodes independently fail on an isolated archive of `89385ff`; 6.08 s. The full declaration module independently reproduces its same three stale scalar-count expectations (53 passed, 3 failed). |
| Affected-domain regression | 1,815 passed, 6 skipped, the same 12 baseline failures, five warnings in 2,873.88 s (exit 1). No new failures or errors; all 551 frozen source/test files unchanged throughout this invocation. |
| Final export/verifier validation | 93 passed, three warnings in 16.80 s (exit 0), including both incomplete-export cases, original embedding owner, compact delivery and ordinary HTTP handoff. All 551 frozen source/test files unchanged. |

Negative coverage uses the new descriptive-comparison and handoff tests alongside
the existing owner, Session, resource and authority suites. These cover invalid
neighbor/Leiden/UMAP values, parameter corrections and zero, missing inputs,
ambiguous/forged references, corrupt accepted evidence, resource mutation and
substitution, foreign revisions, navigation during answer generation, missing
comparison operands, unsupported metric IDs and unqualified prior-output reuse.
The real comparison/reopen drivers forbid scientific calls. Existing model and
feature qualification tests remain authoritative; synthetic metadata comparisons
establish only bounded software behavior, not biological compatibility.

The first new export test invocation exposed an AnnData writer incompatibility
with `numpy.memmap` (six failures). Converting the view with `np.asarray`, as the
existing neighbors owner does, fixed it without copying/densifying a raw matrix.
An initial artifact-audit harness incorrectly called `AnalysisRevision.to_dict`;
it was corrected to serialize the dataclass. Both failed logs remain preserved.

The affected-domain regression extends the exact USI-5 selection with comparison,
Session/history, prior-output, standard-analysis, export/verifier and HTTP tests.
`R/regression.py`, `final-regression-command.json`, JUnit, log and frozen-file
record provide the exact command and result. `R/regression-baseline-comparison.json`
matches the exact 12 failed node IDs to the fresh baseline. Ten failure messages
are identical; the other two differ only in generated definition hashes and
object/context representations, with the same assertions and outcomes.

Final review separately reproduced a `KeyError` for an embedding result containing
`analysis_sha256` without `analysis_path` (`R/partial-export-reproduction.log`).
After the broad run completed, the existing digest checker was changed to catch
that missing key and return `RESULT_IDENTITY_MISMATCH`. Two public-verifier cases
check both incomplete field pairs after proving the complete pair passes. This
exception-handling correction and those tests are the only source/test delta
after the broad frozen run; they change no valid scientific execution or output.
The final targeted run is recorded in `R/final-export-validation*`.
`R/final-validation-delta.json` confirms that only the verifier and its test file
changed between the broad and final targeted snapshots.

The 12 reproduced baseline failures comprise six stale annotation-detail fixture
cases (five dialogue and one interactive), three obsolete scalar-choice counts,
two prerequisite-plan fixtures with obsolete threshold expectations, and one
resource-definition identity fixture. Exact node IDs and tracebacks are retained
in `R/baseline-failures.xml` and `.log`; no expectation was weakened or test
excluded. The baseline archive and current tree use the same Python environment.

`R/preservation.py`, `preservation-after-regression.log` and
`preservation-after-regression.json` verify all
63,908 protected entries unchanged within the initial audit scope, including
size/mtime/mode and the captured small-file hashes. USI-5's 20,938 files and all
1,274 `evals/` files retain their exact file sets. The inherited USI-4 inventory
covers 40,487 files; its 22 pre-existing symlinks were not in that inventory and
are explicitly outside that historical audit claim. Consumed source/model and
auxiliary full hashes match the table above. All operator files recorded at
baseline are unchanged. Native EpiZoo remains clean at
`029cd631d0f9806a646c4a3b42ce10b958f2b67f`. Only the 12 intended tracked files in
section D differ from the initial 708-file tracked snapshot; two tests and this
document are new. The index is empty and HEAD is unchanged.

## J. Final disposition

**USI-6 COMPLETE** within the documented scientific/application scope.

The real ordinary-user analysis, parameter revision, exact cross-Revision
comparison, sibling-branch comparison and downloadable results are demonstrated.
The comparison binds distinct accepted results and reports the real supported
resolution/count changes; it is not merely a historical summary or identity
display. Read-only comparison and fresh-process recovery perform zero guarded
science. Regression reconciliation finds zero new failures. The final narrow
verifier correction passes its affected tests.

The implementation reuses the original catalog resolver, existing bounded
evidence-pair mechanism and compact export path. No scientific algorithm, Registry
tool, authority sidecar, parallel comparison engine or new Session lifecycle is
introduced. Modified-file responsibilities are listed in section D.

USI-7 can proceed with Web presentation work from this scientific/application
handoff, after review. It must retain the documented evidence/reuse distinction,
safe artifact delivery and exact historical bindings. Live-LLM and comprehensive
native-browser qualification remain separate UA3.6.5 work; broader biology and
unsupported quantitative comparisons remain outside this acceptance.

Final `git status --short`:

```text
 M src/agent/application/analysis_delivery.py
 M src/agent/application/dialogue_execution.py
 M src/agent/application/interactive.py
 M src/agent/application/interactive_schemas.py
 M src/agent/application/local_resources.py
 M src/agent/application/scientific_dialogue.py
 M src/agent/application/turn_decisions.py
 M src/agent/orchestration/verifier.py
 M src/agent/tools/analysis/embedding_analysis.py
 M src/agent/tools/analysis/epizoo_embedding.py
 M src/agent/web/static/app.js
 M tests/unit/orchestration/test_verifier.py
?? docs/usi-6-epizoo-standard-analysis-cross-revision.md
?? evals/
?? tests/application/test_standard_analysis_comparison.py
?? tests/web/test_standard_analysis_handoff.py
```

`evals/` was already untracked and is preserved. `git diff --check` passes.
HEAD, local `origin/main` and the final independent remote query remain
`89385ffaffcb3d24ff17a0a0f5d367d07b570fa2`. Nothing is staged, committed or pushed.
