# USI-5 — EpiZoo cross-species and post-training conversational handoff

## A. Baseline and preservation

The verified baseline is `fcfe38cb013f9c63c4e2b9fdbbe80961b39a3a60`,
`Complete USI-4 MAESTRO handoff and shared scientific resources`.
The initial tracked tree was clean; `git status --short` contained only
`?? evals/`. This work is unstaged and uncommitted; no push was performed.

New acceptance evidence is isolated at
`/home/likeyi/agent-usi5-nnqcfsz8` (**R**). The existing material at
`/home/likeyi/program/agent-acceptance/ua3.6.4d-posttraining` (**B**), the
UA3.6.4 final acceptance, USI-4 resource acceptance and `evals/` were protected.
`R/baseline.json` records all 704 tracked file hashes and 42,969 protected file
identities. `R/preservation-audit.json` finds no protected changes: size and
mtime checks cover every recorded file, with content hashes for files at most
2 MiB. This is a bounded preservation audit, not a claim that every historical
large file was independently rehashed. Scientific consumption additionally
checks the original checkpoints, source matrix and resource pins through the
existing owners.

The native EpiZoo checkout remains clean at
`029cd631d0f9806a646c4a3b42ce10b958f2b67f`. Neither native EpiZoo nor
`src/agent/tools/models/epizoo.py` changed. No biological data, weights,
credentials, operator configuration or acceptance workspace is added to Git.
No network resource download was needed.

## B. Existing capabilities reused

The implementation extends the USI-1–4 attachment and typed prerequisite path.
The interpreter selects explicit file roles and scientific declarations. Agent
captures those decisions, preserves the original task, resolves exact registered
resources, and passes a completed input group to the existing scoped Planner,
semantic compiler, preflight, sequential runtime and scientific verifiers.

Reused scientific capabilities are:

- `epizoo_embed_cells` and its native target-inference owner;
- `adopt_scATAC_cell_by_features` and exact sparse matrix conservation;
- `adapt_epizoo_species`, native source/SEAM qualification, native preprocessing,
  native post-training, independent verification and publication;
- `build_cell_neighbors` and `compute_cell_umap`, without requiring Leiden;
- the existing regulatory feature reference owner and FASTA/FAI qualification.

Resource choices use the existing InputSet projection and local resource catalog.
Uploads retain the existing private Session scope. Qualified genome discovery,
explicit permission review, generic scalar parameter binding, accepted outputs,
Session/Revision lineage, evidence, reports and HTTP downloads keep their existing
owners. No scientific tool, workflow engine, checkpoint parser, mapping algorithm,
sharing service or natural-language parameter extractor is introduced.

The conversation's LLM profile remains separate from the EpiZoo scientific model.
The former interprets and plans; the latter is an exact pinned scientific input.
The Registry remains the executable allowlist with 23 tools.

## C. Reproduced gaps and acceptance diagnostics

| Gap or observed failure | Evidence and correction |
| --- | --- |
| Uploaded non-human H5AD lacked a typed EpiZoo declaration/resource handoff | `R/baseline-gaps.log`; extend the existing request/answer role variant, not the scientific engine. |
| Native adaptation resources were omitted from H5AD companion discovery | Admit the existing `adaptation_spec_path` choice through the shared resource continuation catalog. |
| The group source allowlist could not bind EpiZoo input or neutral matrix adoption | Add the two existing source ports and the neutral adoption input's semantic artifact kind. |
| Downstream numeric settings such as UMAP `min_dist=0` did not survive group composition | Derive numeric companion settings from the Registry. The generic parameter binder continues to own validation and literal evidence. |
| An explicit paired model/reference resource could be bypassed by a different dependency in a grouped plan | Validate complete captured paired request resources against the actual plan and resolved execution. A negative test rejects substitution. |
| A Session's newly trained target was not discoverable for the next ordinary uploaded-data request | Project exact accepted current-revision model outputs, and the exact model used by a captured target-inference output, into existing resource choices. No deployment registration or global promotion. |
| Model discovery initially tried to make a legacy adaptation result executable | A new regression reproduced this against a readable result without reusable authority. Discovery now uses existing `optional_binding_for_locator`: unsupported legacy outputs remain readable; corrupt recorded authority still fails closed. |
| Target inference consulted only the external operator model store | Reuse the existing execution-owned `prior_output_store` context for exact local run identity. Only an absent local run may use the explicit external store. Corruption or identity mismatch never falls back. |
| A qualified private genome remained coupled to one peak reference | Derive a Session-scoped genome-only record from the existing owner-qualified reference; retain separate per-task feature binding and all sequence/index/code pins. |

Actual acceptance also encountered legitimate stale historical qualifications.
The original native bundles pin an older adapter implementation, and the older
shared genome record pins an older reference owner. These were not rewritten or
weakened. `R/qualify_native.py` invokes the existing source/SEAM qualification
owners to create new additive bundles under `R/native-bundles/`. The pending
training task explicitly selects the current resource version. The old InputSet
remains captured as history. Resource observations now reject stale native code
before treating a choice as usable.

The stale shared-genome record was preserved. The same FASTA and FAI bytes were
supplemented through normal private uploads and current owner qualification. No
new public provenance or sharing permission was fabricated. The first FASTA
upload hit the existing default 256 MiB upload bound. Its actual 1.71 GB size
justified an isolated acceptance-server setting of 2 GiB, within the existing
supported upload range. No production default, global limit or checkpoint
limit changed.

Failed harness attempts remain in the evidence directory and Session history.
They include use of a nonexistent artifact descriptor field, serialization of a
persisted dataclass through the wrong helper, selection of an optional output
not offered by the output-selection schema, omission of `context=current`, and
use of legacy answer handles without their legacy disambiguation evidence.
The corrected harness uses offered `analysis_path` outputs and modern exact
result-group handles. These attempts do not establish live-provider quality and
did not cause a semantic repair rule to be added to Agent.

An attempted reuse of raw embedding/UMAP outputs correctly found no reusable
schema-2 execution source. Their verification remains readable evidence; it does
not become new reusable authority. The parameter-change witness therefore reruns
the requested inference chain from the same uploaded matrix. A separate witness
then consumes its explicitly selected qualified neighbor graph through existing
prior-output binding, executing only UMAP. This does not expand the scientific
authority contracts.

## D. Implementation and ownership

The modified files and responsibilities are:

| Files | Responsibility |
| --- | --- |
| `src/agent/application/epizoo_handoff.py` (new) | Typed operation/source/species/assembly/measurement/mode declarations; compatible choices; captured model projection; targeted missing roles; task-specific reference binding through existing owners. |
| `src/agent/tools/models/epizoo_inputs.py` (new) | Bounded pinned H5AD/vocabulary/resource observations and delegation to existing scientific qualification. Observations grant no execution authority. |
| `src/agent/application/raw_handoff.py` | Integrate the EpiZoo variant into common capture, supplementary upload, parameter merge, completion and immutable replay checks. |
| `src/agent/application/turn_decisions.py` | Existing request/answer role schema and interpreter instructions for the new typed declaration. |
| `src/agent/application/turns.py` | Offer accessible genome/model resources and route the existing variant. |
| `src/agent/application/local_resources.py` | Existing source-port admission, Registry-derived numeric companions, exact paired resource consumption. |
| `src/agent/application/resource_continuation.py` | Discover existing native adaptation companion resources. |
| `src/agent/application/shared_genome.py` | Private genome-only persistence and scoped reuse; factor existing pin checks without scientific reconstruction. |
| `src/agent/application/reference_supplementation.py` | Filter private genome choices by Session. |
| `src/agent/application/dialogue_execution.py`, `interactive.py`, `session_state.py`, `responses.py` | Existing variant allowlists, pending/replay/display behavior and targeted clarification reason. |
| `src/agent/orchestration/species_adaptation_registry.py` | Existing neutral-adoption source semantic metadata. |
| `src/agent/orchestration/target_model_binding.py` | Exact accepted target resolution through the trusted current RunStore, with bounded existing external-store fallback. |
| `tests/species_adaptation_integration/test_epizoo_handoff.py` (new) | Typed/HTTP continuation, private genome reuse, compatibility, precedence, explicit parameters and fail-closed resource tests. |
| This document | Acceptance and limitations. |

New declaration fields describe actual scientific inputs, not execution steps.
The source H5AD remains attached throughout clarification and supplementary
upload. Explicit values take precedence over captured values; omitted settings
remain scientific-owner defaults. Conflicting generic arguments and scientific
declarations are rejected rather than overwritten. Explicit uploaded FASTA
selection replaces an earlier automatic genome choice only for that role.

Model precedence is explicit valid choice, compatible captured model, then unique
qualified compatible catalog choice. Distinct compatible choices clarify. An
invalid explicit choice never falls through to another checkpoint. Exact alias
definitions can be normalized; species names and filenames do not prove
compatibility. Uploaded checkpoint bytes can identify an already qualified
choice by digest; a loose checkpoint alone does not establish its missing
architecture, vocabulary, corpus statistics or scientific authority.

Captured model choices are transient projections of exact accepted Revision
outputs, not a second persistent model registry. A new target's identity includes
its originating run and step. Existing same-run publication, numerical-runtime,
native-code and checkpoint checks remain mandatory. The store correction adds
no caller-controlled filesystem discovery.

## E. Scientifically supported scope

Real execution uses the preserved **Danio rerio**, taxonomy **7955**, **danRer11**
material from UA3.6.4d. Input semantics are explicitly declared
`binary_accessibility`, not inferred from filename or values. The existing
`B/technical-subset.json` documents the acceptance-only first-16-cell,
observed-region submatrix. USI-5 does not filter a user's matrix automatically.

| Identity | Value |
| --- | --- |
| H5AD | `B/genuine-first16-observed-regions.h5ad` |
| Size | 1,273,712 bytes |
| SHA-256 | `2e76819d2f1ff0a02f04516f9a6a5e14e1a3ab66a1663a40237224568f3d2883` |
| Sparse matrix | 16 cells × 17,922 features; int64 CSR; 19,959 nonzeros |
| Ordered cell-axis digest (newline-delimited) | `c61642a5a3acafa545a652d9c69c5f8483d47d23f4176d2bac8e528926facb84` |
| Target feature-axis digest | `fef49956379c596be63e08aa8a195ec7c234efa8f2b2bd8530c09e56e56002a8` |
| Native source checkpoint SHA-256 | `6b2d13fdbd54a9b0d56efa5afa81bc4832b813f4eac8662e4d93cc08c9d9b39a` |
| SEAM checkpoint SHA-256 | `d40b5626622a302f693f2e36ad4f4942e83cf6daf602e1add5216f72b75e2530` |

Target inference requires the accepted model's exact feature membership and
measurement semantics. The existing owner handles supported order changes and
records their provenance; no silent vocabulary intersection or feature dropping
is introduced. It does not request FASTA, FAI, SEAM or mapping for inference.

Native `de_novo` training uses the exact uploaded feature coordinates and a
qualified matching genome; it does not need a target-to-source mapping. The
existing `mapped_reference` mode still requires qualified mapping, source
vocabulary, chain and pinned mapping tools. Binary mapped-reference input is not
silently accepted. No new real mapped-reference run or new species qualification
is claimed here.

The existing native training default remains 500,000 steps. Acceptance explicitly
requests **2 steps**, learning rate **1e-5**, batch size **1**. Native seed 0,
float32 parameters, CUDA float16 autocast, sequence batch size 4, maximum sentence
length 8192 and the remaining owner defaults remain recorded. Invalid explicit
steps/rate/batch values fail Registry validation. Zero UMAP distance is retained.

## F. Real inference and continuation

`R/real_workflow.py` uses the normal HTTP upload/turn path and actual
`InteractiveAgentApplication`. Language interpretation, capability selection,
planning and output selection are scripted; scientific inference is real.
The first request resolves the unique compatible previously accepted model,
manifest SHA-256
`49aaac55da9189ce55039f21dcfbb214991005af786a4ec3672c0ee5a6ccbf77`.

Its plan is exactly embedding → neighbors → UMAP. Inference runs once
(approximately 3.20 seconds), with zero native training, mapping or SEAM calls.
Revision:
`dce112c4bd777d2bca362bcd9c90219f8c51c99a62a3dbeb587e4e15e4534589`.

The real downloadable compact H5AD contains finite 16 × 512 EpiZoo embeddings
and finite 16 × 2 UMAP coordinates in exact source cell order. No Leiden output
is present. Initial downloads include:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `analysis.h5ad` | 98,472 | `f19f6295f0d0e72340b859890ddc908af1d504a0070a442696b291d3f63af093` |
| `neighbors.h5ad` | 90,544 | `28e4b38c859331ab54e243c2f20f4d41f5f165b8fe5a94538e1db99737d3c92d` |

The target model has accepted schema-2 adaptation authority; the embedding result
has its existing successful verification and exact target provenance. The graph
has its existing reusable schema-2 authority. This does not claim a newly added
schema-2 embedding or UMAP authority.

`R/continuation.py` preserves the original uploaded source and model and executes
an explicit new request with **5 neighbors** and **min_dist 0**. Resolved arguments
retain both values. The resulting revision is
`8369370c021540055128f4834b58f21c1d104b17d307bb0b8386543b7cd7909b`.
The earlier revision remains identical. `R/neighbor_reuse.py` then uses the
qualified current neighbor graph with **min_dist 0.1**, executing only
`compute_cell_umap`; no EpiZoo inference, training, mapping or SEAM call occurs.
Its revision is
`c2d8be6c5829500dc2ac39f98129a861de4f43dee8edf3a0004856d5487cffdb`.

Fresh-process reads restore results and download their artifacts with providers
and instrumented scientific owners forbidden. `R/artifact-audit.json` independently
reopens downloaded H5AD files, checks exact ordered cells, finite embeddings and
coordinates, absence of Leiden, and exact sparse source conservation for the
adopted matrix. Restart download comparisons check byte identity.

## G. Real native post-training

The training request captures the original matrix, explicit two-step budget,
learning rate and batch size. A missing strategy clarifies. Supplementary native
resource selection and genome uploads resume that same task. Source identity and
numeric overrides remain captured across failed admissions and process restarts.

Actual plan: `adopt_scATAC_cell_by_features` → `adapt_epizoo_species`.
Revision:
`b33ee261aa9c02b649d0cb85c72d8bbb335683f66dd079735bb9ccb6ea8cd557`.
The native backend executes SEAM/sequence preparation once (134.64 seconds) and
training once (7.81 seconds). The native checkpoint is independently verified;
subsequent result/evidence verification calls are counted separately in
`R/training-science.jsonl`. The instrumented driver records five reference
build/FASTA verification calls and nine target-verifier calls across execution
and result composition; it does not misreport these as one total owner call.

| Native execution fact | Accepted value |
| --- | --- |
| Method | `epizoo-native-posttraining.v2`, `de_novo` |
| Trainer attempts / optimizer updates | 2 / 2 |
| AMP-skipped steps / finite-loss steps | 0 / 2 |
| Last total / SR / CCA loss | 7.076188087463379 / 5.944897174835205 / 1.1312909126281738 |
| Device | NVIDIA GeForce RTX 4090 |
| Peak allocated / reserved GPU bytes | 5,992,334,336 / 6,297,747,456 |
| Training matrix identity | `4e507d26fd8ccde8a3c5392449fabf4d9031d3ec4dc1d5eb1940c6b1e48aebc4` |
| Reference identity | `92d9e55082a35852d17a2aef2907a38f6314a9a0c5c9e6796925e62cf2669a03` |
| Published manifest SHA-256 | `e91a4fdef31aae18c386f25b0633b945a2b8485911e372c9429ee9f0a9f89143` |

Actual HTTP downloads:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `epizoox-checkpoint.pth` | 1,262,255,722 | `e130786f5a904988bb7d926fb2b06f50c100f8f8914e91904bb8e37ef3bb63a6` |
| `epizoox-model.json` | 3,250 | `2456b0428888c6d49b4de73465dac0fab187581614607818532e1c38c2850a38` |
| `matrix.h5ad` | 1,067,628 | `dc8d5e02910a164e680322bb2ba25c48c76ac116db462acb34ce7442971c19ce` |

The existing checkpoint-specific **5 GiB** delivery bound is unchanged. The
metadata identifies architecture, native execution settings, vocabulary and
corpus-statistics digests, actual updates and reuse limitations. Checkpoint plus
metadata remains explicitly **not a standalone portable inference bundle**.
No training loss is represented as biological performance or convergence.

A subsequent ordinary request uploads no replacement model and selects no
internal model ID. The captured newly trained target takes precedence over the
also-available historical compatible target. Real inference loads the new model
and runs once, with zero training/SEAM/mapping calls. This supplies an additional
native serialization/loading witness. The child revision is
`95cdd69892b28b3f8544b712448c00c67a8756c46d5577d3aae08a176021fff4`.
The consumed model run/step and manifest are exact; the training revision and
previous checkpoints are unchanged. After the legacy-discovery correction,
`R/captured-model-audit.json` rechecks both exact accepted revisions through the
final projection and priority logic, with zero science/provider construction;
the private genome is absent from foreign-Session offers.

`R/training-reopen.log` and `R/trained-reuse-reopen.log` record fresh-process
restoration and downloads with zero instrumented science and no provider calls.
Recovery here means completed-publication and application recovery, not a newly
qualified mid-training optimizer/RNG restart.

## H. Resource lifecycle, explanations and privacy

Uploaded matrices and new weights remain private. The model is accessible through
its Session/Revision, not automatically shared. Captured model discovery never
searches another Session. Qualified genomes are offered only if deployment-wide
review permits reuse or the private record belongs to the same Session.

The synthetic HTTP test qualifies an uploaded FASTA, restarts the application,
then binds a different exact peak set against the persisted private genome
without reuploading sequence/index bytes. A foreign Session is rejected. Existing
reviewed-shared genome tests continue to exercise deployment reuse. This software
scope is distinct from the real run's private uploaded genome qualification;
the stale historical shared record was not claimed as a newly valid shared proof.
Per-task BED/vocabulary identity remains distinct from sequence/index identity.

The user-facing path needs files, scientific declarations and necessary choices.
Internal reference manifests and digests are constructed by Agent and scientific
owners. This does not make an unqualified raw checkpoint or arbitrary mapping
scientifically usable. The acceptance deployment supplies already-qualified
source/SEAM resources through existing operator configuration; production use
must explicitly deploy the new qualified bundle identities if desired. No
production configuration or default resource was changed.

`R/result_review.py` reads accepted evidence and report/figure artifacts over HTTP,
then obtains ordinary evidence-grounded answers with scripted modern result-group
selection. Training answers cite the exact historical model result, two attempts,
two optimizer updates, 16 cells, 17,922 features, native method, `de_novo` and
`not_assessed` biological performance. Inference answers cite the exact current
embedding result, preserved cell order, 512 dimensions, danRer11 and binary
semantics. Hashes and detailed configuration remain available in accepted evidence
and download metadata; no unsupported improvement claim is introduced.

The read witnesses cover four training-Session output projections and four
report/figure downloads, and five inference-Session output projections and six
report/figure downloads before the final graph-only continuation. Generation-
checked navigation to the earlier revision and back preserves every Revision.
All covered EpiZoo/reference scientific call counts are zero during these reads,
answers and navigation. Failed scripted answer attempts remain visible separately.
This is neither cross-Revision comparison work nor live-language qualification.

The final `R/final_read_guard.py` witness adds fresh-process execution and
reconstruction guards: run/resume, the executor, step/run verification,
`VerificationContext.verify`, evidence/report rebuilding, provider construction
and the covered EpiZoo/reference owners are forbidden. **19 training-Session
reads and 32 inference-Session reads pass**, including revision/evidence views,
artifact inventories and non-checkpoint downloads. All **77 training-workspace
files and 67 inference-workspace files** retain their exact size/mtime identities
and hashes for files at most 2 MiB. The large checkpoint download was already
verified twice and is not redundantly transferred by this final witness. See
`R/training-final-read-guard.json` and `R/inference-final-read-guard.json`.

## I. Tests and regression

Validation uses `/home/likeyi/anaconda3/envs/agent/bin/python -B`, `PYTHONPATH=src`
(and the existing native EpiZoo checkout only for real native witnesses). Acceptance
limits CPU worker threads to four and uses the existing numerical environment.
All `RUN_*` test gates are disabled. Test logs, JUnit reports, exact commands and
frozen source/test/static/resource hashes are under R. Historical `evals/` are not
collected. No tests are deleted or scientifically weakened.

The final focused invocation passes **159 tests, two warnings in 167.51 seconds**,
exit 0, including the legacy-authority discovery correction. All **549 frozen source/test/static/resource files** are unchanged.
`R/final-focused-command.json`, `final-focused.xml`, `final-focused.log` and
`final-focused-result.json` retain the exact command, individual results and
freeze comparison. The warnings are the existing pkg_resources and Louvain
deprecations. The final affected-domain invocation finishes with **1,456 passed, six skipped,
twelve baseline failures and five warnings in 1,758.94 seconds**, pytest exit 1.
All **549 frozen files** remain unchanged. The 47 selected paths collect 1,474
cases with no individual exclusions. `R/regression-command.json`,
`regression.xml`, `regression.log` and `regression-result.json` retain the exact
invocation and outcomes. This is not an all-green suite: the twelve failure names
exactly match the combined baseline evidence, with **zero new failures**
(`R/regression-baseline-comparison.json`).

The six skips are the explicit Chromap qualification gate, three native EpiZoo
preprocessing cases, the large CPU checkpoint-load case and CUDA parity. Their
`RUN_CHROMAP_QUALIFICATION`, `RUN_EPIZOO_INTEGRATION`,
`RUN_EPIZOO_CHECKPOINT_TEST` and `RUN_EPIZOO_PARITY_TEST` gates remain disabled.
The separate real USI-5 execution witnesses do not relabel these skipped tests
as passed. Warnings concern the existing pkg_resources/Louvain deprecations,
an intentional duplicate-observation fixture, JUnit record_property formatting
and the installed TBB version disabling Numba's TBB threading layer.

The first affected-domain attempt was interrupted after reproducing the new
legacy-discovery defect: 718 passed and ten failed in 740.98 seconds (exit 2).
Its failure set contains the nine known baseline failures and the added
`test_same_dialogue_across_capabilities[adapt_epizoo_species-values2-fields2]`
failure. All 549 frozen files were unchanged. Evidence is retained under
`R/interrupted-before-legacy-correction/`. After the correction, the entire
scientific-dialogue test file plus the EpiZoo handoff tests pass: **75 passed,
two warnings in 11.87 seconds** (`R/legacy-discovery-correction.log`). The full
affected selection is rerun without exclusions; the interrupted run is not
claimed as completed regression acceptance.

Development results, which overlap and are not additive: initial relevant batch
120 passed; private-genome/reference batch 18 passed; expanded EpiZoo/adaptation
batch 214 passed. The handoff file has 13 tests after explicit captured-model and
local-store checks. The earlier focused invocation passed 159 tests in 128.13
seconds with all 549 frozen files unchanged; the final repeat above also covers the captured-inference-model projection
and legacy-authority discovery correction.

The affected-domain selection extends all 29 USI-4 regression paths with native
adaptation, target inference, legacy EpiZoo, model delivery, generic parameters,
resource identity, decisions, prior-output authority and relevant web paths.
The nine documented USI-4 baseline failures are retained for exact comparison:
six annotation fixtures omit the already-required `arguments` key; three expect
obsolete scalar totals. The expanded selection additionally exposes three
pre-existing failures outside that earlier selection. All three independently
reproduce on a pristine archive of the verified HEAD, whose 704 file hashes
match the initial baseline:

- `test_definition_identity_captures_every_operator_field_and_is_detached`
  includes absent `fastq_whitelist=None` in its expected digest, whereas the
  existing definition deliberately omits that absent extension to preserve
  legacy identities (one failed in 1.55 seconds).
- The two C1 variants in `test_prerequisite_planner_handoff.py` expect the old
  missing-threshold parameter specification/call count. The pristine baseline
  returns the same differing specification and four rather than three calls
  (two failed, 11 passed in 2.20 seconds).

See `R/baseline-extra-provenance.json`, `baseline-extra.xml`,
`baseline-prerequisites.xml` and their logs. These are twelve baseline failures
in the expanded selection, not grounds to weaken current scientific checks.

Real evidence is separate from synthetic software fixtures. Real science used
scripted language decisions, the actual HTTP/application/compiler/runtime and
native scientific owners. No live LLM or native-browser qualification is claimed;
UA3.6.5 remains responsible for that scope. No new provider-specific rules,
UI redesign, arbitrary-species support or new mapped-reference qualification is
included.

Reproduction entry points (run creation scripts only in a fresh isolated workspace;
existing turn IDs are immutable):

```bash
PY=/home/likeyi/anaconda3/envs/agent/bin/python
R=/home/likeyi/agent-usi5-nnqcfsz8
PYTHONPATH=src "$PY" -B "$R/regression.py" --focused
PYTHONPATH=src "$PY" -B "$R/regression.py"
# Read-only restoration of existing acceptance:
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
PYTHONPATH=src:/home/likeyi/program/EpiZoo \
"$PY" -B "$R/real_workflow.py" --training --reopen
```

## J. Final disposition

**USI-5 COMPLETE**, within the stated **Danio rerio / danRer11, binary
accessibility, native de_novo technical acceptance scope**. Ordinary uploaded
input handoff, real inference, bounded real native training, model and analysis
downloads, captured-model reuse, explicit parameter changes, qualified-graph
continuation, grounded evidence answers and fresh-process recovery have passed.
The new legacy-discovery regression is corrected; the remaining twelve affected-
domain failures are independently documented baseline failures.

**USI-6 can proceed** from this bounded integration result. This does not qualify
biological improvement, production convergence, arbitrary species/vocabularies,
new mapped-reference inputs, standalone checkpoint portability, mid-training
crash resume, live LLM interpretation or native-browser usability. Existing
scientific dependencies and public/private resource permissions remain required.

USI-6 cross-Revision comparison, USI-7 UI redesign and UA3.6.5 integrated live
product qualification have not begun. Nothing has been staged, committed or
pushed.

Final `git status --short` (`evals/` was already untracked):

```text
 M src/agent/application/dialogue_execution.py
 M src/agent/application/interactive.py
 M src/agent/application/local_resources.py
 M src/agent/application/raw_handoff.py
 M src/agent/application/reference_supplementation.py
 M src/agent/application/resource_continuation.py
 M src/agent/application/responses.py
 M src/agent/application/session_state.py
 M src/agent/application/shared_genome.py
 M src/agent/application/turn_decisions.py
 M src/agent/application/turns.py
 M src/agent/orchestration/species_adaptation_registry.py
 M src/agent/orchestration/target_model_binding.py
?? docs/usi-5-epizoo-cross-species-posttraining-handoff.md
?? evals/
?? src/agent/application/epizoo_handoff.py
?? src/agent/tools/models/epizoo_inputs.py
?? tests/species_adaptation_integration/test_epizoo_handoff.py
```
