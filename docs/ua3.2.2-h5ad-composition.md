# UA3.2.2 — Registered H5AD scientific composition

This implementation connects one registered H5AD, explicit typed scientific
declarations and exact reviewed EpiZoo resource pins to the existing interactive
scientific execution path. The Interpreter selects operations; the ordinary
Planner and semantic compiler construct the actual plan. Uploading remains a
byte-transfer/registration operation and runs no providers or scientific tools.

## Baseline and scope

Implementation starts from `main` at
`a72aa31ffb5f660bca9585dfe5869b14934ed9c5` (`Complete UA3.1 web H5AD upload and
multi-turn interaction`). Root acceptance records the actual branch/reference
checks, source freeze and preserved evaluation/output inventories. No commit or
push is performed for UA3.2.2.

The existing scientific path remains:

```text
Completed upload resource ID + typed companion InputSet + reviewed resource choice
→ LocalResourceAdmission.compose_h5ad / existing RegisteredInput
→ immutable interactive turn / existing Interpreter and Planner
→ semantic compiler and compiled-plan source/resource admission
→ registered owners / normal verification
→ accepted evidence, report and Session Revision
```

The application adds no execution recipe. Inspection-only, direct embedding
and the existing inspection → embedding → neighbors → Leiden → UMAP semantic
chain remain independently planner-authored choices. The existing semantic
channels, scientific algorithms, numerical defaults and cache lifecycle remain.
Historical graph/embedding scientific-authority expansion, general multi-H5AD
composition, conversational parameter extraction and UI redesign are deferred.

## Pretrained foundation policy

The project owner's final UA3.2 closeout instruction approves the original
EpiZoo pretrained weights as the default foundation for supported, qualified
species. EpiZoo uses one jointly pretrained human/mouse checkpoint with distinct
species-specific vocabulary, frequency and filter resources. The human input
axis has 1,355,445 features and 700,460 retained cCREs; the mouse input axis has
1,341,077 features and 814,020 retained cCREs. Species selection changes the
qualified auxiliary configuration, not the checkpoint binary. The approved
mouse configuration below binds the existing checkpoint
`6b2d13fdbd54a9b0d56efa5afa81bc4832b813f4eac8662e4d93cc08c9d9b39a`
to its exact species-specific auxiliary resources. Read-only review confirms
this identity equals M14's established pretrained source identity. Approval
does not qualify unfamiliar files or another species by implication; UA3.2
configuration does not replace M14's independent source-bundle qualification.
The original EpiZoo repository remains clean at
`029cd631d0f9806a646c4a3b42ce10b958f2b67f`; the preserved M14 source bundle
`b2077733fd0f9e2fb770f918980236bda81f7c4a7cfaa0f64fbc1cd69d5d0c35`
binds the same pretrained checkpoint. This is a read-only consistency check,
not a new M14 execution or resource qualification.

Only mouse receives an automatic default in this closeout. Human source/vocabulary
audit identities match the preserved source bundle, but human direct-inference
acceptance is absent and the recorded frequency/normalization concern remains
unresolved in [the interface audit](epizoo_interface.md). Mouse acceptance does
not qualify that additional direct-inference configuration.

Ordinary embedding uses the original pretrained weights directly, without
fine-tuning, post-training or adaptation. General task-specific fine-tuning is
future work: it must be explicitly requested, initialize from a qualified source,
preserve the original weights, and publish separately qualified derived weights
with exact source-model and training lineage.

Existing `adapt_epizoo_species` remains unchanged. It consumes its independently
qualified pretrained source bundle, SEAM sequence resources, target reference
and explicit strategy. `mapped_reference` inherits qualified mapped cCRE
parameters and retains the existing unmapped-parameter initialization contract.
`de_novo` initializes target-specific parameters without coordinate correspondence
while preserving applicable pretrained shared parameters. Both perform qualified
target-species post-training and publish a separate target checkpoint; neither
means training the entire model from scratch. Source/SEAM identities, target
species/assembly/vocabulary, execution profile, binary-accessibility restrictions
and existing training/verification limits remain authoritative. No strategy is
automatically chosen. General adapted-target-model inference remains deferred.

## Operator configuration and interactive use

An operator may designate an existing InputSet as an H5AD companion. Such a set
contains declarations and supported optional parameters; the registered source
owns its path. A companion cannot provide a second H5AD source, override the
registered input or contain arbitrary execution fields. Resource pins are
configured separately from companion declarations and planning-model profiles.

The existing Web configuration accepts the following illustrative configuration:

```json
{
  "workspace_root": "/operator/agent-workspace",
  "default_profile_id": "lab-model",
  "model_profiles": [
    {
      "profile_id": "lab-model",
      "provider_id": "openai",
      "model_id": "<operator-admitted-model-id>",
      "display_label": "Lab planning model"
    }
  ],
  "upload_root": "uploads",
  "input_sets": [
    {
      "input_set_id": "mouse-science",
      "label": "Mouse scientific declarations",
      "h5ad_companion": true,
      "execution_inputs": {"species": "mouse"}
    }
  ],
  "epizoo_resources": [
    {
      "resource_id": "mouse-recorded-qualified",
      "label": "Reviewed mouse EpiZoo resources",
      "species": "mouse",
      "checkpoint_path": "/operator/reviewed/pretrained_EpiZoo.pth",
      "checkpoint_sha256": "6b2d13fdbd54a9b0d56efa5afa81bc4832b813f4eac8662e4d93cc08c9d9b39a",
      "frequencies_sha256": "c4c63aaae8a6f841812189bf59930014162c3f61c7a463d744be71595359171d",
      "filter_indices_sha256": "96a80287ae085d7e9e10d05f0dd7d5b266ad86d08b91b01ccc07af6b9b01e393",
      "qualification": "Recorded EpiZoo interface and preserved Fang2021 smoke acceptance; UA3.2.2 real 32-cell integration",
      "default": true
    }
  ]
}
```

The original pretrained mouse default is approved by the project owner's final
closeout instruction. The example does not activate a running deployment. Its
mouse content identities come from the recorded acceptance below; replace the
workspace/checkpoint locations and planning model with actual operator-admitted
values. When adding these fields to an existing operator JSON, preserve unrelated
workspace, provider and input-set fields. Provider credentials remain in the
existing server environment. Activate the intended configuration from the
repository root with `PYTHONPATH=src python -m agent.web --config
<operator-config.json>`. No scientific source edit is required.

The approved default requires an explicit `mouse` declaration and the exact
reviewed resources. A configured label or path does not qualify unfamiliar files.
Auxiliary resources
continue to use the existing scientific backend's fixed species-specific resource
layout. No arbitrary auxiliary-directory setting is added.

The browser selects a completed upload, an admitted scientific companion and,
when needed, a reviewed EpiZoo resource. It submits opaque identifiers:

```json
{
  "turn_id": "analysis-1",
  "expected_generation": 0,
  "utterance": "Compute EpiZoo embeddings, neighbors, Leiden clustering and UMAP.",
  "resource_id": "local-<registered resource digest>",
  "input_set_id": "mouse-science",
  "epizoo_resource_id": "mouse-recorded-qualified"
}
```

Explicit valid resource selection takes precedence. Omitting it selects only a
uniquely applicable resource explicitly configured as the qualified default for
the declared species. Missing/ambiguous resource selection is checked when the
actual plan uses embedding. Invalid explicit selection fails without fallback.
Inspection needs neither species nor EpiZoo resources. Ordinary noncompanion
InputSets retain their existing exclusion with uploaded-resource selection.

Natural language selects scientific operations through the existing LLM
contracts. Species and newly supplied numeric parameters come from typed
companion inputs; mentioning `mouse` or `resolution 0.7` only in an utterance does
not establish those execution arguments in this stage. Parameters with shared
names use the existing compiler's explicit scope, such as
`neighbors_random_seed`, `cluster_random_seed` and `umap_random_seed`.

## Source and resource integrity

The composed binding preserves the resource ID, registration digest, immutable
source locator and captured source bytes. Existing integrity checks run before
new consumption. Independent `execution_inputs` cannot be combined with a
registered binding.

At the existing preexecution acceptance hook, the application checks actual
compiled H5AD consumers. Direct consumers must bind the exact registered source;
same-plan embedding may consume the existing validated inspection output from
that source. Substitution, unsupported lineage or a changed binding fails before
science. Unrelated operations are not required to consume the upload. This uses
the existing plan/reference contracts and adds no scientific lineage system.

New registered embedding executions carry an optional public
`expected_resource_identity` with exactly `resource_id`, `checkpoint_sha256`,
`frequencies_sha256` and `filter_indices_sha256`. The embedding owner checks
current checkpoint/auxiliary bytes and compares the cached model's recorded
checkpoint digest before inference. For a cold load, the checkpoint is
streamed into a private snapshot and its exact copied bytes are hashed before
the existing model loader consumes that snapshot with memory mapping. This
protects all new cold backend loads. Auxiliary
files are hashed and parsed from the same byte payload. A warm cached model with
a different digest fails closed, even when its path/device/dtype cache key
matches. Owner-produced actual resource identities and paths must match the
captured pins and current fixed layout; resources are checked again before
artifact publication.

Pin failures have bounded safe codes/messages. They expose no server paths.
Existing scientific owners still validate sparse counts, supported species,
feature positions/order, cells, checkpoint structure, runtime and device.
Resource pin checks establish invocation provenance, not independent inference
reproduction or broader historical scientific authority.

## Defaults and provenance

Omitted optional parameters retain existing owner behavior. Valid explicit typed
values are passed and invalid values fail. There is no application/Web default
table, biological calibration or automatic adjustment for small datasets.

| Owner | Existing optional defaults retained |
| --- | --- |
| EpiZoo embedding | `device="cuda:0"`, `overwrite=False`; qualified registered execution supplies its exact checkpoint |
| Neighbors | `n_neighbors=15`, `metric="euclidean"`, `random_seed=0`, `overwrite=False` |
| Leiden | `resolution=1.0`, `random_seed=0`, `overwrite=False` |
| UMAP | `min_dist=0.5`, `spread=1.0`, `random_seed=0`, `overwrite=False` |

The embedding wrapper retains batch size 4, maximum length 8192, random sampling,
seed 0, requested AMP, FP32 cached parameters, zero workers and disabled progress
display. Pinned executions publish bounded owner-produced
`epizoo-resource-provenance.v1`: expected/actual resource identities, species and
effective inference settings including overwrite. Registry/verification checks
this projection against resolved pins; evidence explicitly admits the reviewed
optional projection. Historical unpinned results retain their required fields
and remain readable with provenance absence distinct from successful pin checks.
Downstream owner results and compact H5AD provenance retain effective scientific
parameters; omitted request arguments remain distinguishable from defaults.

## Validation and scientific acceptance

Software integration uses provider-neutral scripted Interpreter/Planner/output
selection through the ordinary Web/application interfaces. Only EpiZoo model
loading/inference is replaced in these checks. The public embedding owner,
downstream Scanpy owners, Registry, verifier, evidence/report and Revision
acceptance run normally. Coverage includes direct embedding and the existing
five-tool chain, default/explicit parameters, prerequisites, inspection-only,
source/resource checks and completed question/retry/restart history without
scientific replay. The affected focused regression results are recorded after
the final source freeze; no full repository suite or live provider is invoked.

Final affected regression: **1,151 passed, 3 skipped, 3 warnings in 254.46
seconds**, exit 0. All inherited `RUN_*` gates were removed, bytecode and pytest
cache writes were disabled, and tests used an isolated `/tmp` base directory.
The skips are the existing opt-in real EpiZoo owner gates. Warnings concern the
existing Scanpy/louvain deprecations and unavailable newer TBB threading layer.
The command uses the existing `agent` environment. The portable command below
retains the exact test target list; its isolated temporary-directory basename
is replaced with a placeholder:

```text
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider \
  --basetemp <isolated-test-directory> \
  tests/web \
  tests/application/test_h5ad_composition.py \
  tests/application/test_local_resources.py \
  tests/application/test_registered_resource_integration.py \
  tests/application/test_fastq_resources.py \
  tests/application/test_interactive_boundary.py \
  tests/tools/analysis/test_epizoo_embedding.py \
  tests/tools/analysis/test_embedding_analysis.py \
  tests/tools/models/test_epizoo.py tests/tools/models/test_epizoo_cache.py \
  tests/unit/orchestration/test_registry.py tests/unit/orchestration/test_verifier.py \
  tests/unit/orchestration/test_error_policy.py \
  tests/report/test_evidence.py tests/report/test_analysis_report.py \
  tests/unit/orchestration/test_semantic_compiler.py \
  tests/unit/orchestration/test_semantic_registry_metadata.py \
  tests/unit/orchestration/test_semantic_prompt.py \
  tests/unit/orchestration/test_semantic_wire_v4.py \
  tests/unit/orchestration/test_llm_planner.py \
  tests/unit/orchestration/test_llm_planner_v4.py \
  tests/unit/orchestration/test_llm_planner_v4_recovery.py \
  tests/tools/data/test_scatac.py
```

All 430 frozen source/test/static/resource files remain byte-identical during
this final run; aggregate freeze SHA-256 is
`015d2fc13c4c9423701f2ef0c927d6ab8e000b0fc4272540e3d76aa21c22974d`.
The actual v3 schema assertion now includes the new optional input-only resource
pin binding; existing planner behavior is unchanged. Python 3.10 lightweight
Registry imports remain supported, and pinned provenance preserves explicit
`cpu:0` rather than altering an existing device declaration.

All 705 pre-existing evaluation files and 198 preserved output files retain
their names, sizes, modification times, modes and content digests. Their repeated
inventory/content aggregate SHA-256 values remain
`049386e611cbc0f1d176d36b1a0079a54d67d2efde634f6326021f25b729038a`
and `37f9a1e1bdad27c4b7485c0f07ff6b7c7bb33d94a1cab8375e2e3dc5e2acfa3d`,
respectively. Existing operator configuration and real web state are untouched.
At the UA3.2.2 checkpoint, `main`, HEAD and local `origin/main` remained at the
baseline and the index was clean. The implementation comprised 22 modified
tracked files and four new documentation/test files, all unstaged. No fetch,
pull, reset, commit or push occurred. UA3.2.2 implementation and bounded acceptance
were ready for review; current closeout status follows below.

Ten native Firefox checks pass in an isolated server/browser workspace: actual
file selection and upload registration, upload/companion coexistence in both
selection orders, default/explicit resource controls, ID-only composer payloads
and safe switching to ordinary InputSets. Composer POSTs are intercepted before
interpreted submission for these transport/UI checks; providers, science and
interpreted turns remain zero, with no JavaScript errors. This browser check is
separate from the normal HTTP/application execution tests and real science below.
The original `native-checks.json` and screenshot remain outside Git. The native
record's SHA-256 is
`f8d58e971a337ac052576b307a49bd98f94e48209e20632c7858fd391c715acc`.

Read-only resource/runtime readiness checks found the previously qualified mouse
checkpoint and both auxiliaries byte-identical to
[the recorded EpiZoo interface](epizoo_interface.md) and preserved
`outputs/epizoo_smoke/Fang2021_2000_metadata.json`:

| Input/resource | Current SHA-256 |
| --- | --- |
| Original Fang2021 2,000-cell H5AD | `16189d0dabd21cc1d9f51e5f39662da82037fe9f54938f8593a3ff49aa50cfab` |
| EpiZoo checkpoint | `6b2d13fdbd54a9b0d56efa5afa81bc4832b813f4eac8662e4d93cc08c9d9b39a` |
| Mouse frequencies | `c4c63aaae8a6f841812189bf59930014162c3f61c7a463d744be71595359171d` |
| Mouse filter indices | `96a80287ae085d7e9e10d05f0dd7d5b266ad86d08b91b01ccc07af6b9b01e393` |

The available `agent` runtime is Python 3.11.16, PyTorch 2.7.1+cu128,
FlashAttention 2.8.3, AnnData 0.12.19 and Scanpy 1.11.5. EpiZoo core imports and
CUDA visibility pass on the RTX 4090 (24 GB class).

One bounded real integration check uses the original source's first 32 ordered
cells and complete 1,341,077-feature axis, created sparsely in an isolated
temporary directory. The actual Web upload/composed-input path executes all
five existing production owners once, with scripted provider-neutral language
interfaces and real EpiZoo inference. Every step passes existing verification;
the embedding is `(32, 512)`, UMAP is `(32, 2)`, Leiden returns four clusters,
and every cell remains in exact source order. The accepted Revision, evidence,
safe report and figure are available. The persisted scientific run lasts 21.29
seconds; the embedding step lasts 13.30 seconds. The original source, checkpoint
and auxiliary hashes remain exact before/after execution and recovery.

A later accepted-result question and completed retry execute zero science.
Reconstructed-server reads/retry/evidence/report run with all provider/scientific
callbacks forbidden and preserve all 25 existing acceptance-workspace files.
The original harness checks peak allocated GPU memory below 24 GiB. Its final
summary serialization then fails because a frozen provenance mapping was not
converted to JSON; the accepted run and all preceding assertions had completed.
The corrected summary is recovered read-only in a separate process with no
scientific replay. The numerical peak is unavailable because it was not
persisted before that reporting failure; no exact peak value is claimed.

The real acceptance workspace, datasets, scripts and original logs remain
outside Git. This document is the durable non-sensitive summary. The recovered
`acceptance.json` SHA-256 is
`57aa4be923706bbf660cc97897e03a24134ab21b3b3fc6f6d93bf6f8019eb5e9`;
the original reporting-error log SHA-256 is
`8c92ae98a02d9790ce6ca76de3299edb7c2d1ef5341752141abf37e8afc2b93f`.
Recovered stdout has the same digest as the recovered JSON. No datasets,
scientific outputs or private environment locations are copied into this record.

This qualifies bounded registered-H5AD integration on this reviewed mouse
source/resource/runtime, with the existing inference-verification limits. It
does not establish additional model-quality or biological-generalization claims.

Production operator configuration remains an operator-owned selection. Broader
model-quality/biological evaluation, live-provider language qualification and
historical embedding/graph reuse remain outside this stage.

## UA3.2 review and final closeout

UA3.2.3 continues from the same committed baseline and preserves the unstaged
UA3.2.2 implementation. Existing software, native-browser and real scientific
acceptance were inspected without repeating the full focused suite, live
providers or GPU inference.

A new bounded read-only check reopened the preserved normal application with
provider and Registry scientific callbacks forbidden. Session and question
reads, completed retry, evidence, report and both PNG handles passed. Normal
RunStore reads retain all five passed step verifications. Artifact reads confirm
finite `(32, 512)` embeddings, finite `(32, 2)` UMAP, four clusters, exact source
cell order and sparse neighbor graphs. Captured resource pins equal the owner's
actual/expected provenance and evidence projection. All 26 current acceptance
root files remain byte-identical before/after this check: 24 workspace files,
the subset and recovered summary. The original 25-file assertion preceded
creation of the summary. The delivered safe report SHA-256 remains
`4848ad1d226e3fde19267211d69644efa219392bc658faf3c05cc0440ee88492`.

Scientific review reproduced a resource-read attribution defect: a temporary
resource mutation around the actual read could escape content-pin checks after
the bytes were restored. A filesystem-stat-only guard also failed when mutation
and restoration occurred within the same timestamp tick. The correction binds
consumed content directly: cold checkpoint loading streams a private
snapshot, hashes its exact copied bytes, and uses the existing memory-mapped
model load; auxiliary byte payloads are both hashed and parsed through
`BytesIO`. Consumed identity does not rely on filesystem timestamps. Scientific
algorithms, numerical defaults and model-cache keys remain unchanged.
The safeguards apply to new execution. Existing provenance schema/field sets
and accepted records remain unchanged; historical pinned records are not
retroactively claimed to have used the new consumed-byte safeguards.

A cold load needs scratch disk sufficient for the checkpoint-sized
snapshot. CPU memory mapping retains the unlinked snapshot's backing storage
until model/cache references release it. A read-only capacity check confirms
available scratch space exceeds the 5.23 GB checkpoint; it does not copy the
real checkpoint or run inference. The accepted real run remains preserved, with
no GPU inference replay.

The directly affected checks pass, exit 0 in both runs:

| Targeted validation | Result |
| --- | --- |
| Embedding owner, model and cache | 75 passed, 3 existing opt-in skips, 8.12 seconds |
| Web scientific integration | 10 passed, 3 existing warnings, 12.72 seconds |

Portable equivalents of the exact targeted commands, with the original unique
temporary-directory names replaced by placeholders:

```text
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider \
  --basetemp <isolated-owner-test-directory> \
  tests/tools/models/test_epizoo.py \
  tests/tools/models/test_epizoo_cache.py \
  tests/tools/analysis/test_epizoo_embedding.py

PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider \
  --basetemp <isolated-web-test-directory> tests/web/test_h5ad_analysis.py
```

Inherited `RUN_*` gates are disabled before execution. The owner run explicitly
unsets `RUN_EPIZOO_INTEGRATION`, `RUN_EPIZOO_CHECKPOINT_TEST` and
`RUN_EPIZOO_PARITY_TEST`; its three existing real-science skips remain expected.

The combined result is 85 passed and 3 skipped. All 430 frozen
source/test/static/resource files remain unchanged across both runs; the freeze
SHA-256 is
`5553568fff154388f59965e89198685cda22cd635a7dc0298892a917aff4d607`.
Neither run invokes a live provider or real GPU inference. The full repository
suite and original 1,151-test suite are not repeated.

The earlier UA3.2.3 review left default approval and a deployment target
unresolved. The project owner's final closeout instruction now approves the
original pretrained foundation as the default for supported, qualified species
and explicitly separates software acceptance from deployment activation. The
reviewed mouse checkpoint and auxiliary identities therefore support the
`default: true` example above. Current integrity findings match the recorded
mouse identities. No arbitrary local model or unfamiliar species is approved by
this policy.

Valid explicit compatible resources take priority over the uniquely applicable
configured default. Missing, ambiguous, changed or incompatible resources fail
closed; an invalid explicit choice never substitutes a default. Human direct
inference qualification is assessed separately from mouse acceptance and M14's
pretrained-source qualification; no human automatic default is configured.

**ACCEPTED — UA3.2 software and scientific integration complete.** Registered
H5AD composition, ordinary semantic execution, exact consumed-resource identity,
provenance and historical compatibility pass their scoped acceptance. Final
read-only pretrained-source consistency, species qualification and actual
configuration checks pass. The final configuration/API run passes **70 tests
in 2.54 seconds**, exit 0, with no production/source/test changes in this final
turn. All 430 frozen files retain the prior
`5553568fff154388f59965e89198685cda22cd635a7dc0298892a917aff4d607`
freeze digest. Current pretrained checkpoint and both mouse auxiliary content
identities match the recorded hashes; subsequent checks preserve their complete
stat fingerprints without a redundant large-file rehash.

Portable final targeted command, with all inherited `RUN_*` gates disabled:

```text
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider \
  --basetemp <isolated-final-test-directory> \
  tests/web/test_config.py tests/web/test_h5ad_composition_api.py
```

The established operator configuration is now configured and validated with
`upload_root: "uploads"`, the `mouse-science` companion and the approved
`mouse-recorded-qualified` default. Existing provider, PBMC input and planning
default fields are preserved; a secure original backup stays outside Git.
The resulting operator JSON SHA-256 is
`dd86afd663fc2345b542bd78fa4bf97a5ebbfaf6d2557e84736bcddb4a6d4389`.
The existing configuration loader/resolver validates explicit priority, invalid
explicit selection without fallback, species mismatch, missing human default
and ambiguous-default failure. These checks invoke no providers or scientific
execution. No server is started: **configured and validated; server startup
required**. The placeholder example above supplies portable activation guidance.

Final closeout starts with HEAD, local `main`, local `origin/main` and the
independently inspected remote `main` at
`a72aa31ffb5f660bca9585dfe5869b14934ed9c5`, preserving 24 modified tracked files
and four new documentation/test files with a clean index. Earlier UA3.2.3 remote
checks matched the same baseline at both review start and final verification.
Repeated full inventory/content checks preserve all 705 evaluation files and
198 scientific output files, with the same aggregate digests recorded above.
The project owner authorizes one coherent closeout commit and normal push when
all acceptance requirements pass:
`Complete UA3.2 H5AD composition and pretrained EpiZoo integration`.

Species and new numeric values still require typed companion inputs. General
historical embedding/graph/clustering authority reuse, scientific H5AD/table
downloads, FASTQ/BAM/fragments Web upload, cCRE reference defaults, automatic
peak projection and assembly conversion remain deferred.
