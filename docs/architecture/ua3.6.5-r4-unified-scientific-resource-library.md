# UA3.6.5-R4 — Unified scientific resource library

2026-10-19. Initial R4 acceptance is retained below; the final architecture closeout supersedes its interim review disposition.
Evidence root **R**: `/home/likeyi/agent-ua365-r4-2qttwv3k` (outside Git).

## R3 closeout and protected baseline

All 17 R3 review files matched the recorded final hashes. The prior focused and
frozen-regression evidence remains applicable, including its documented corrected
test assertion. HEAD, main, origin/main and remote main initially matched
`6c15055623877f48f8ed7dedf69b1c783ebe5a68`; the index was empty and only historical
`evals/` was untracked. The explicitly authorized scoped commit and normal push
produced **`5e5555302ea401b18328ad412238b4da6e13daaf`**, “Complete UA3.6.5-R3 native
scientific integration simplification”. R4 starts from that commit. No force push,
reset or cleanup was used.

R/baseline.json inventories **171,293 protected entries**, extending the R3
inventory with the completed R3 acceptance and current normal workspace. Protection
uses size/mtime/mode plus SHA-256 for files at most 64 KiB and all historical evals;
it is not a new full-byte audit of every large dataset/checkpoint. The original
operator bytes are retained in R/operator-before.json. New normal Sessions and the
explicit resource-configuration change are distinguished from historical data.

## Responsibility changes

| Before | R4 implementation |
| --- | --- |
| Common references registered into the task upload catalog, then separately promoted | One `ScientificResourceLibrary` in the existing `shared_resources.py`; installation writes one persistent `index.json`. No private registration or promotion is required for installed references. |
| Three promotion writers (`shared_resources`, `shared_genome`, `reference_supplementation`) | All three removed. Explicit user sharing and operator installation use the same library installation method. |
| Permission-file coverage, pinned legal evidence and rereading that evidence on lookup | Removed from current installation/discovery/consumption. A short provisioning usage basis records the intended use; source terms are reviewed operationally, not interpreted by the runtime. |
| Separate genome discovery projection | Folded into shared discovery. Library genomes feed the existing genome prerequisite and exact task-feature binding. They do not invalidate the generic pending catalog. |
| Reference usability coupled to annotation/genome/QC implementation-file hashes | New application resource records pin scientific bytes only. Historical readers ignore only the exact known implementation identities, preserving scientific file checks. Result authority and native model requirements are unchanged. |
| Common EpiZoo models listed in normal operator JSON | The two exact existing model declarations are indexed in the same library and supplied to the existing native model-selection interface. No model is retrained or copied. |
| Ordinary uploads potentially confused with reusable references | Upload registration remains private task storage. Installation supports only explicit annotation, genome, QC and qualified EpiZoo roles; H5AD/BAM/FASTQ/fragments are rejected as library kinds. |

`LocalResourceAdmission` remains the upload identity/source-binding owner. It is
not the writer for new shared library entries. Its private scientific records
retain uploaded task references and required source/feature lineage. Small old
record readers preserve accepted Sessions and explicit historical references;
old promotion and legal-evidence writers are absent. No historical result or
private reference is silently migrated into the library.

`ScientificInputSet` remains an internal typed projection used by existing
selection/compilation and captured historical bindings. Users do not construct it.
Static task input mappings, native QC reference contracts, scientific result
publication/authority and EpiZoo vocabulary validation have actual distinct
consumers and remain. This change replaces application shared-resource management,
not the scientific owners' reference formats or native algorithms.

## One library and its use

The normal root is **`/home/likeyi/program/agent-scientific-resources`**, separate
from Git, task storage and protected acceptance workspaces. Its index records
resource kind, identity, label, release/version, scientific resource description,
provenance, intended usage, exact file paths/checksums/sizes. Annotation specifications
use the existing owner's format under `annotations/`; installed source files and
models in stable operator locations are indexed in place. Explicitly shared uploads
receive library-owned copies, as detailed in the final closeout. The library creates no Run, receipt, authority or
promotion object. Existing native scientific specifications are not duplicated
into a second scientific schema.

Installation invokes the corresponding existing owner. Reinstalling an unchanged
resource returns its original entry without repeating scientific qualification.
Index writes are locked and atomic. Resource versions coexist; an indexed file is
never overwritten by installation. Discovery reads index metadata; selection
checks exact file integrity. A checksum supplies identity, not biological validity.
Native execution retains its relevant method-specific checks.

Genome FASTA/FAI can be installed directly with `install_genome`, without a dataset,
prior Run or fabricated counting history. Exact sequence/index validation reuses
the existing streaming owner. When an analysis needs a feature reference, its
exact ordered feature axis is bound privately to that genome by the existing
scientific owner. Sharing the genome does not share the experimental feature BED.

Complete uploaded annotation references remain private by default. The existing
Session settings now list that Session's reusable references and offer an explicit
“Share for future laboratory analyses” action. The matching facade/HTTP operation
requires explicit shared intent and the originating Session. It accepts no arbitrary
path or experimental upload. No reference-file download endpoint is added. Repeated
sharing is idempotent and does not overwrite the private source or its history.

Compatibility and ambiguity remain owned by the existing typed handoff. MAESTRO
requires matching species/assembly, method, measurement, namespace and tissue
context. Different meaningful candidates prompt a choice. Captured selections
remain exact across continuation; discovery cannot replace uploaded inputs.

## MAESTRO provisioning and usage basis

The existing MAESTRO/R/Presto environment is reused unchanged. Fresh primary-source
review verifies the pinned MAESTRO **1.5.1**, revision
`74f10babf4125b98ef005df4186b14ab02e13bfd`, package DESCRIPTION/license, exact bundled
`human.immune.CIBERSORT.RData`, and its annotation tutorial. R/source-review.json
records URLs, publisher, release, acquisition time, bounded download sizes and
SHA-256. R/signature-check.log independently compares the installed TSV with the
RData: all **1,319 candidate/gene pairs, 28 candidates**, exact order and strings;
only the TSV header differs.

The [pinned MAESTRO tutorial](https://github.com/liulab-dfci/MAESTRO/blob/74f10babf4125b98ef005df4186b14ab02e13bfd/example/ATAC_infrastructure_10x/ATAC_infrastructure_10x.md)
explicitly uses this packaged marker list for immune annotation. Its
[package metadata](https://raw.githubusercontent.com/liulab-dfci/MAESTRO/74f10babf4125b98ef005df4186b14ab02e13bfd/DESCRIPTION)
identifies GPL >=3. The CIBERSORT authors describe academic non-profit use in their
[methods publication](https://pmc.ncbi.nlm.nih.gov/articles/PMC5895181/).

The operational assessment is that the requested **local non-commercial laboratory
research computation**, using the actual MAESTRO-packaged marker list as documented,
has a sufficient provisioning basis. This is a scoped inference from the package's
explicit use instructions and the authors' research-use statement, not a new
third-party redistribution or unrestricted hosted/commercial-service grant.
CIBERSORTx website-service terms are not treated as the terms of every MAESTRO data
file. The CIBERSORT deconvolution engine or website is not invoked. Reference bytes
remain server-local; no alternative marker panel or scoring semantics is substituted.

The indexed scientific resources retain these exact identities:

- GRCh38 refgenes: SHA-256 `9b18e56531155226450254aefd192bb7f49d03d74c5bbfdc53af5411a0d2fb37`.
- Human immune marker TSV: SHA-256 `3c202410d4f46049a298784838e5f9d214fc67fcddf34536d1c3811e383b601a`.
- The two previously configured human/mouse EpiZoo model/vocabulary declarations,
  unchanged; exact entries and pins are in the library and R/provisioning.json.

The normal operator change adds `scientific_resource_root` and removes its duplicate
`epizoo_resources` list after indexing those same models. All other configuration
fields and credentials remain unchanged. Before/after operator SHA-256:
`f59312491c2719ae93bc3366724335550a713e5c76cf6bd84ddbc01a0558477e` →
`b95d6331a07f8aa19a18de5a57fd2f2183293b0e876489a5584bbd14c5f12150`.
The library currently contains three common entries. Source references, models and
checkpoints were not modified or copied into Git.

## Validation

The first resource-focused development run had 28 passes and one incorrect test
assertion: exact aliases are intentionally equivalent, so an ambiguity case must
use distinct resource identities. The corrected case checks distinct identities.
The genome-integration development run exposed dictionary-label projection and
pending-catalog integration defects. These were fixed at the shared projection;
no scientific algorithm or language workaround was added. The subsequent focused
run passes **37 tests in 57.37 seconds**, with two existing dependency warnings.

The frozen adjacent regression passes **922 tests, seven warnings, 1,033.31 seconds**,
exit 0. All **557 source/test/static files** remain unchanged throughout that run
(R/regression-result.json and R/regression.xml). RUN_* gates are disabled, with no
exclusions from the selected modules. It covers annotation integration, resource
selection/continuation, uploads, Web configuration/delivery, Session/dialogue reads,
EpiZoo handoff and adjacent PCA/neighbor scientific owners. This is an affected-domain
regression, not a claim that the entire repository suite was rerun.

Three subsequent bounded fixes extend polling to answered/clarification/navigation
until their captured-profile presentation exists, expose private genome references
through their existing Session scope, and validate newly installed EpiZoo checkpoints
with the native shape/vocabulary owner using CPU memory mapping. The first final
affected-module run passes **134 tests, two warnings, 50.98 seconds**. A later
private-genome listing test accidentally added a second candidate to an existing
execution fixture and correctly caused clarification (26 passes, one failure);
the listing assertion now has its own fixture. The final resource/private-scope
rerun passes **28 tests, two warnings, 60.15 seconds**, exit 0
(R/scope2-tests.log).
These checks are reported separately from the earlier frozen 922-test run.

Both real installed human/mouse checkpoints also pass native shape/vocabulary
validation, 793 tensors each, with no training, inference or GPU allocation
(R/model-verification.json). Native EpiZoo source and R3 scientific algorithms are
unchanged. UA3.6.5 remains open; annotations are not independent biological validation.

## Real normal laboratory Web acceptance

The real server loads `/home/likeyi/program/agent-web-config/operator-web.json`
and `/home/likeyi/program/agent-web-workspace`, using the existing Conda/R runtime.
Firefox accesses the ordinary Web UI on the isolated acceptance port 8874. The
provider is real OpenRouter `openai/gpt-5.6-sol:floor`, through the normal
`openrouter-gpt-5-6-sol` profile. Acceptance instrumentation observes calls; it
replaces neither provider responses nor scientific functions. No acceptance-only
reference configuration is supplied.

Two new ordinary Sessions upload the authentic source H5AD, request annotation in
natural language and answer necessary assembly, measurement and grouping questions:

| Session | Accepted revision |
| --- | --- |
| `c800af0d-aa41-434f-b09a-42238ebe2627` | `054801ea6138ee3889324f1ee8ab480aee9a658d0241c14235137e71b6cc058b` |
| `3d96a708-1ee5-40cd-a9b7-ad5f3c657e51` | `b11612317a9b89f7c9425c6f79fd9c57d35f8598f72694755957c6abe7019725` |

Neither Session uploads or registers the common reference. Both capture the exact
library identity `b59229c31a7c6e6735930c98b200f5a40a004c9013e4a95103a4276d`.
The source is the preserved real **1,004 × 82,579** PBMC H5AD, SHA-256
`c75bca64515710f6045a1c65d9f0dfb9fe9827102501856007b2bfcb58b33c81`, with
8,164,121 sparse nonzeros and 22,279,127 supplied counts. No feature filtering,
relabeling, rescaling or matrix densification is performed.

Each Session executes native MAESTRO once and independently replays verification
once. Across the two runs, call accounting records four native annotation calls,
two verifier calls and **zero library annotation requalification calls**. The native
regulatory-potential matrices and cell assignments match exactly between Sessions;
the library's three entries and index bytes are unchanged. Native RP has shape
1,004 × 28,307 with 12,960,672 nonzeros, and matches the retained full R2B native
diagnostic exactly. All six groups / 1,004 cells receive computational assignments;
zero cells are unassigned or ambiguous. Biological validity remains `not_assessed`.

The UI exposes the summary, annotation rationale and method limitations. The
scientific report opens successfully. Native Firefox downloads the annotated H5AD:
79,569,234 bytes, SHA-256
`bb2f118cf38d30bd1631d66bee0ffd1aad00123febc18d2f26c86a433f26f759`.
Independent sparse-array checks preserve exact X dtype/values, cell and feature
order, original metadata and per-cell group-label propagation. Delivery correctly
removes private source-path metadata. Export bytes differ from the original
scientific artifact by this reviewed delivery projection.

Participation evidence remains explicit: 31 peaks / 6,062 counts do not match the
reference; 3,286 additional peaks / 238,421 counts receive zero transcript weight.
The output does not claim that every supplied peak contributed. Report, download,
refresh and follow-up reads execute no annotation science. Detailed evidence:
R/normal-annotation.json, R/second-annotation.json, R/two-session-reuse.json,
R/science-check.json, R/native-downloads.json and native Firefox captures.

A live follow-up exposed two different behaviors. The answer provider used
`groups_omitted` (display truncation) to support a claim about unidentified groups.
The existing evidence guard rejected that explanation and published the safe
accepted-facts fallback; no semantic retry or scientific re-execution occurred.
This limits qualification of fluent live-provider answers. Separately, the browser
could stop polling at `answered` before that immutable presentation arrived. The
final UI fix and its checks are recorded below; the accepted scientific result
and evidence guard remain unchanged.

## Historical access and preservation

Fresh read-only processes open preserved USI-4, R2B and R3 Sessions and accepted
results. Their respective 38, 71 and 43 workspace files remain byte-identical;
provider construction and scientific execution are zero. No migration or historical
record rewrite occurs. Evidence: R/historical-read.json,
R/r2b-historical-read.json and R/r3-historical-read.json (with corresponding scripts
and logs). The final broader preservation check is recorded below.


## Final recovery, operation and limitations

Native Firefox executes the actual polling predicates for all three delayed
dialogue statuses, including completed and historical no-profile cases (nine
status cases). These are labelled simulated lifecycle probes, separate from
real-provider acceptance. A new live follow-up reaches a complete displayed
answer with no transient client error. Its generated prose is rejected for an
unbound factual literal and the accepted-facts fallback is displayed correctly.
Both live follow-up limitations remain visible; no provider-quality fix or semantic
retry is claimed. Across the complete acceptance there are 14 provider calls.
Scientific call totals remain the two productions and two independent replays.

A fresh Web process loads the same normal configuration and restores both Sessions,
including exact history, generation, active revision and revision lists. Summaries
and reports render after recovery. All **173 normal workspace files** remain
byte-identical, with **zero provider calls and zero scientific calls** during
recovery (R/recovery.json). The final 171,293-entry protected inventory comparison
reports no unexpected changes; only the explicitly provisioned normal operator
configuration differs, with its original bytes retained. No historical Session,
accepted scientific artifact, checkpoint, reference source or eval is rewritten.

Ordinary users of this configured laboratory Web can now annotate the compatible
human PBMC data with the common reference without uploading or privately
registering it. Other species, tissue contexts and methods still need appropriate
resources; incompatible or meaningfully ambiguous candidates are not silently
selected. QC installation currently supports the existing qualified human/mouse
bundle; arbitrary QC annotation parsing is not introduced. The final closeout makes model choices read the current library index alongside
annotation/genome/QC discovery. Static task/parameter bindings remain supported;
legacy inline model catalogs and reference-only operator catalogs are retired.
The normal deployment's common references/models live only in the library.

The usage assessment covers the requested local non-commercial research deployment.
It does not grant unrestricted hosted/commercial use or reference redistribution.
No new raw-reference download route exists. Library indexing cannot prevent an
external operator from mutating a referenced file; selected-file integrity checks
fail closed if that occurs. Experimental uploads cannot be installed as library
kinds. Private reference sharing requires an explicit action in the originating
Session; it does not silently broaden other private task records.


## Initial review disposition (superseded by final closeout)

R4 is ready for scoped review and remains **unstaged, uncommitted and unpushed**.
HEAD, local main, origin/main and the queried remote main remain the R3 commit
`5e5555302ea401b18328ad412238b4da6e13daaf`; the Git index is empty.
`git diff --check` passes. Only the reviewed source/test/static/documentation changes
and the new R4 report are candidates for a future scoped commit. Historical
`evals/` stays untracked and untouched. No credential, configuration, binary,
checkpoint, private source or acceptance evidence is staged. Temporary acceptance
Firefox and Web processes are stopped; the persistent normal configuration and
library remain installed.

The concrete deletions are the three promotion writers, current permission-evidence
construction/checking and separate genome-discovery function. New shared entries
bypass the local upload catalog entirely. Retained components have distinct owners:
private upload identity and Session isolation, exact task-feature binding, historical
readers, internal typed planning projections, and native scientific validation and
result authority. The change adds the library/index and explicit sharing UI, so
total line count grows; the simplification is the removal of duplicated shared
resource phases and responsibilities, not a claimed net reduction in source lines.

## Final architecture closeout — durable sharing and current consumers

Closeout evidence root **C**: `/home/likeyi/agent-ua365-r4-closeout-aktakh5h`.
The starting branch is `main`; HEAD, local main, origin/main and queried remote
main all equal `5e5555302ea401b18328ad412238b4da6e13daaf`. The index is empty.
All 18 initial R4 review-file hashes match R/final-git.json. The only untracked
items are the R4 report and preserved `evals/`. The normal operator still selects
`/home/likeyi/program/agent-scientific-resources`, containing the same installed
annotation and two native model entries. No operator or credential change is
required for this closeout. C/baseline.json extends protection over the completed
initial R4 evidence, normal workspace and library.

**Durable uploaded references.** Explicit sharing now calls the same library
`install` with owned-file copying. Required components are streamed into immutable
content-addressed locations under the existing library root, with exact size and
SHA-256 checks, independent source inodes, space checks and fsynced publication.
The source is never moved or rewritten. Annotation copies refgenes, signatures
and any declared QC scope; QC copies its source annotation, TSS, lineage, native
parent reference and that parent's complete required FASTA/FAI/cCRE/annotation
closure. Genome sharing copies FASTA/FAI only, excluding the experimental feature
BED. Native contracts validate reconstructed library-local manifests; portable
scientific identities and values stay exact. QC manifest identity changes where
its native contract binds relocated parent-manifest bytes; accepted originals do
not change. Trusted operator-installed files and large models remain indexed in
place.

All files must exist and pass integrity/qualification before the single index is
atomically replaced. Repeating an unchanged sharing operation reuses the same
copies and entry without repeating scientific qualification. Interrupted imports
leave no discoverable incomplete entry; completed unindexed copies can be reused
by retry. This is a bounded addition to existing file/index primitives, not a
second transaction, transfer, catalog or authority system.

**Current consumer inventory and retirement.**

| Owner / consumer | Final responsibility |
| --- | --- |
| `shared_resources.ScientificResourceLibrary.install` | Only current common-reference installer and index writer; operator indexing and explicit uploaded-file sharing converge here. |
| `shared_resources.discover` → `turns.respond` / annotation and QC handoff | Current common choices come only from the library. The same projection may add the current Session's private qualified references. Old deployment records are no longer discovered. |
| `reference_supplementation.choices/select` → genome prerequisites; `epizoo_handoff.prepare` | Library genome selection plus exact Session-private feature binding. Old global genome/combined-reference discovery branches are removed. |
| `InteractiveAgentApplication` → Web choices, workers and typed model admission | Model choices are read from the library. Inline operator model catalogs, constructor/turn overrides and Web-owned model catalog injection are removed. The launcher no longer forwards a second catalog. |
| `WebConfiguration` / `ScientificInputSet` | Configured library root plus explicit task/parameter templates. Reference-only annotation/QC and legacy genome catalogs are rejected with an installation instruction. Internal typed projections and exact scientific argument bundles remain; they do not publish shared records. |
| `LocalResourceAdmission.register` | Private uploads and task-reference identity only. New legacy deployment scientific/genome/combined-reference registrations are explicitly rejected. |
| `shared_resources.persist`, `shared_genome.persist_private/bind`, `reference_supplementation.qualify` | Retained private scientific-upload and exact feature-lineage owners, used by annotation/EpiZoo/FASTA handoff. They cannot publish new shared deployment records. |
| Historical validators / exact-reference readers | Read immutable prior records and captured bindings, check scientific bytes, and preserve accepted result access. No promotion, global rediscovery or legal-evidence workflow is restored. |

The three promotion writers, permission-evidence machinery and duplicate genome
entry-point removed in initial R4 stay deleted. Closeout additionally deletes the
inline model configuration/parser and Web model-catalog state, retires global
legacy discovery and blocks its remaining generic registration entry point.
C/legacy-inventory.json identifies 111 persisted legacy records in protected
acceptance evidence, supporting bounded read compatibility. Historical record
schemas and accepted authority are not rewritten. Scalar/model tuples passed
inside typed compilation are transient projections, not another persistent store.
Model discovery now observes the library directly rather than requiring a restart
for newly indexed choices. Existing exact continuation/catalog checks still reject
changed captured selections.

Tests use clearly labelled metadata-only model fixtures where inference is mocked
or withheld; they are not resource qualification or live-provider evidence. The
obsolete launcher/catalog-structure assertions are removed. Lifecycle tests cover
private original use, independent copies, full multi-file closure, restart,
new-Session consumption after original-file loss, idempotence, corruption,
interrupted copy/index publication, private isolation and experimental-file rejection.
All destructive negative checks use isolated fixtures, never historical directories.

**Final real product verification.** The normal operator, native Firefox and real
OpenRouter `openai/gpt-5.6-sol:floor` create Session
`289679ed-c548-493d-a8a1-14906870215e`. After the factual X/group/assembly reply,
native annotation accepts Revision
`9f7a7714ab2fd589ada739ba7101b5fc230905812d44189e533b566a6a95dd3d`.
The same installed reference is captured without upload or requalification.
Five actual provider calls lead to one production and one independent native
replay. Library qualification calls are zero; the three-entry index SHA-256 stays
`29e85ad296e3fb1de3dcaff17c2f380996206f074eff633328c636a220275e91`.
Native cell assignments match the prior R4 result exactly; RP again matches the
full R2B diagnostic with zero sparse differences. Existing biological limitations
and feature participation remain unchanged.

Firefox opens the report and downloads the annotated H5AD, 79,569,234 bytes,
SHA-256 `c94c784c53ed55d99015fdcb25fc82fbeb295526b926ba0b455acb1998ef763f`.
The new run's provenance changes its artifact/export byte identity; source X,
axes, original metadata and propagated labels remain exact. Download inspection
confirms exclusion of private source-path metadata. After a fresh service process,
Session history, generation, active Revision, summary and report restore, and the
H5AD delivery digest remains identical. All **211 normal workspace files** stay
byte-identical across read-only recovery; provider and scientific call counts are
zero, including comparison with the pre-restart audit digests. C/science-check.json,
C/native-reuse-check.json, C/native-downloads.json and C/recovery.json record these
checks. Native Firefox screenshots were inspected. Existing R4 two-Session evidence
is reused for the unchanged native algorithms; it is not recounted as new science.

USI-4, R2B and R3 historical readers again preserve all 38, 71 and 43 files,
respectively, with no provider construction or scientific work. The closeout
protection inventory covers **191,897 files** and the post-regression comparison reports no changes,
including operator configuration, installed resources, earlier Sessions/evidence,
models and `evals/`. This retains the stated bounded size/mtime/mode plus small-file
and eval SHA policy; it is not a new full-byte checksum of every large protected file.

**Frozen regression.** The final snapshot contains 557 source/test/static files,
manifest SHA-256 `ed6a986fa4d472c9b39c321d7c41e7ba04f6e812f3026a39046ce5fd4a5bb112`.
C/frozen-source.json records every file SHA-256; the runner verifies the full file set
and contents after each process. Focused and broad scopes overlap and are not added.
The broad regression comprises two disjoint process scopes, retaining every selected
module. No timeout, assertion or production behavior is changed for the baseline
rich-API timing limitation described below. All `RUN_*` environment gates are removed;
there are no exclusions within the selected modules.

| Final scope | Result | Exit |
| --- | --- | --- |
| Focused resource/consumer and corrected baseline assertions | 147 passed, 3 warnings in 158.35 s | 0 |
| Instrumented rich-API module | 7 passed, 1 baseline-reproduced polling-deadline failure in 223.96 s | 1 |
| Remaining affected modules | 1,473 passed, 2 baseline-reproduced artifact-inventory assertion failures, 7 skipped, 8 warnings in 2,054.77 s | 1 |

The two disjoint broad scopes contain 1,490 cases: **1,480 passed, 3 failed,
7 skipped**. C/final-test-inventory.json verifies distinct case identities and records
each failure/skip. The focused 147 passes overlap and are not added to this count.
All seven skips require the explicitly disabled `RUN_CHROMAP_QUALIFICATION` gate.

This is not an all-green broad regression. The rich-API failure is the
unchanged 20-second test polling deadline during presentation finalization after
successful science. The same case fails on committed R3 (42.68 s complete test);
standalone current behavior and diagnostics do not identify a resource-library or
scientific regression. It remains a baseline test/runtime timing limitation. Native
Firefox product/recovery acceptance uses the real browser and passes independently.

The other two failures are unchanged result-delivery assertions in
`test_unified_scientific_result_delivery.py`. Both reproduce on committed R3 in
26.35 s (exit 1): the actual inventory also contains the child revision's own accepted
`clusters.h5ad`, which the older expected filename sets omit. Scientific execution,
evidence, and report/figure checks preceding those assertions pass. The tested
production delivery modules and these two tests are unchanged by R4. These baseline
assertions remain visible; the suite is not relabelled as passing.

Exact commands and JUnit records are in C/focused-command.json, C/rich-regression-command.json,
C/regression-command.json and the corresponding XML files. The repository working
directory uses `PYTHONDONTWRITEBYTECODE=1`, `PYTHONPATH=src` and no `RUN_*` variables:

```bash
/home/likeyi/anaconda3/envs/agent/bin/python -B -m pytest \
  tests/annotation_integration/test_shared_resources.py \
  tests/web/test_reference_supplementation.py \
  tests/web/test_config.py \
  tests/web/test_species_handoff.py \
  tests/species_adaptation_integration/test_epizoo_handoff.py \
  tests/web/test_leiden_resolution.py::test_unique_legacy_literal_normalizes_offsets_before_verified_execution \
  tests/web/test_multi_source_selection_transport.py::test_selection_transport_rejects_invalid_or_unregistered_inputs \
  tests/web/test_multi_source_selection_transport.py::test_transport_retains_submitted_order_without_assigning_scientific_roles \
  -q \
  -o \
  cache_dir=/home/likeyi/agent-ua365-r4-closeout-aktakh5h/focused-cache \
  --basetemp=/home/likeyi/agent-ua365-r4-closeout-aktakh5h/focused-temp \
  --junitxml=/home/likeyi/agent-ua365-r4-closeout-aktakh5h/focused.xml
/home/likeyi/anaconda3/envs/agent/bin/python -B -m pytest \
  tests/web/test_rich_api.py \
  -q \
  -o \
  cache_dir=/home/likeyi/agent-ua365-r4-closeout-aktakh5h/rich-regression-cache \
  --basetemp=/home/likeyi/agent-ua365-r4-closeout-aktakh5h/rich-regression-temp \
  --junitxml=/home/likeyi/agent-ua365-r4-closeout-aktakh5h/rich-regression.xml
/home/likeyi/anaconda3/envs/agent/bin/python -B -m pytest \
  tests/annotation_integration \
  tests/application/test_local_resources.py \
  tests/application/test_registered_resource_integration.py \
  tests/application/test_resource_choice_identity.py \
  tests/application/test_fastq_resources.py \
  tests/application/test_uploads.py \
  tests/application/test_fastq_uploads.py \
  tests/application/test_bam_uploads.py \
  tests/application/test_fragments_uploads.py \
  tests/application/test_interactive_session_core.py \
  tests/application/test_dialogue_detail.py \
  tests/application/test_dialogue_evidence.py \
  tests/species_adaptation_integration/test_epizoo_handoff.py \
  tests/tools/analysis/test_pca_embedding.py \
  tests/tools/analysis/test_neighbors_publication.py \
  tests/application/test_human_epizoo_mapping.py \
  tests/application/test_initial_scientific_declarations.py \
  tests/application/test_string_scientific_execution.py \
  tests/application/test_scientific_parameter_continuation.py \
  tests/web/test_api.py \
  tests/web/test_attachment_role_handoff.py \
  tests/web/test_bam_chat_handoff.py \
  tests/web/test_bam_execution.py \
  tests/web/test_bam_selection.py \
  tests/web/test_chat_attachments.py \
  tests/web/test_config.py \
  tests/web/test_external_fragments_execution.py \
  tests/web/test_fastq_chat_handoff.py \
  tests/web/test_fastq_execution.py \
  tests/web/test_fastq_selection.py \
  tests/web/test_fragments_chat_handoff.py \
  tests/web/test_fragments_selection.py \
  tests/web/test_h5ad_analysis.py \
  tests/web/test_h5ad_composition_api.py \
  tests/web/test_launch.py \
  tests/web/test_leiden_resolution.py \
  tests/web/test_model_profiles.py \
  tests/web/test_multi_input_scientific_roles.py \
  tests/web/test_multi_source_selection_static.py \
  tests/web/test_multi_source_selection_transport.py \
  tests/web/test_output_selection_api.py \
  tests/web/test_qualified_resource_choices.py \
  tests/web/test_qualified_resource_handoff.py \
  tests/web/test_raw_resource_resolution.py \
  tests/web/test_reference_supplementation.py \
  tests/web/test_resource_continuation.py \
  tests/web/test_rich_static.py \
  tests/web/test_role_clarification_scientific_continuation.py \
  tests/web/test_scientific_artifact_static.py \
  tests/web/test_scientific_artifacts.py \
  tests/web/test_scientific_binding.py \
  tests/web/test_selection_clarification.py \
  tests/web/test_selection_defaults.py \
  tests/web/test_session_upload_reuse.py \
  tests/web/test_species_handoff.py \
  tests/web/test_standard_analysis_handoff.py \
  tests/web/test_static.py \
  tests/web/test_target_resource_continuation.py \
  tests/web/test_unified_scientific_result_delivery.py \
  tests/web/test_upload_api.py \
  tests/web/test_upload_static.py \
  tests/web/test_uploaded_multiturn.py \
  -q \
  -o \
  cache_dir=/home/likeyi/agent-ua365-r4-closeout-aktakh5h/regression-cache \
  --basetemp=/home/likeyi/agent-ua365-r4-closeout-aktakh5h/regression-temp \
  --junitxml=/home/likeyi/agent-ua365-r4-closeout-aktakh5h/regression.xml
```

Development results are distinct: the first consumer run exposed a temporary
comprehension typo; corrected consumer checks then exposed old path-equality
assertions and a test helper assuming `Path` rather than accepting a string from a
fresh process. A copy-interruption fixture passes; the new index-interruption
fixture initially supplied an intentionally incompatible base reference and failed
qualification before reaching its intended interruption. It now uses an actually
qualified private reference. The corrected fixture/fresh-process/launcher checks
pass 11 tests. A preliminary freeze was interrupted to retire a stale launcher
assertion; another focused checkpoint had 136 passes and this one fixture failure.
These development logs remain in C and are not presented as final all-green evidence.

The production changes are confined to application `shared_resources`,
`shared_genome`, `reference_supplementation`, `local_resources`, `interactive`,
`epizoo_handoff`, `turns`; Web `config`, `app`, `__main__`, `static/app.js` and
`static/index.html`. Relevant fixtures/tests, README, AGENTS and this report are
updated. No scientific algorithm, Registry contract, EpiZoo backend, native R
script or accepted scientific authority schema changes. The scoped file list and
final hashes are recorded in C/final-git.json.

The Answerer is deliberately unchanged. Lexical constraints, evidence-only fallback
and the observed `groups_omitted` semantic misuse remain a possible next focused
UA3.6.5 work unit; no English-word or MAESTRO-specific prose patch is introduced.
The final resource model is one index/installer, owned copies for shared uploads,
indexed stable operator resources, private task storage and bounded historical
reading. Source-use terms remain a provisioning responsibility. Temporary
acceptance processes are stopped at closeout; normal configuration and library
remain ready for ordinary use.

Final review also closed explicit legacy-handle consumption: old deployment
annotation/QC handles and old global genome bindings are rejected for new execution,
even when supplied directly. Their metadata/content readers remain usable. An earlier
broad run was deliberately stopped after 899 passes (exit 2), with no source changes
during that run; it is not a final result. Two focused historical-read/current-selection
rejection checks pass. Final native recovery was repeated on the corrected production
source; the library branch and scientific algorithms used by the fresh live annotation
are unchanged.

The subsequent `fd0f24f4df2f154ec624cc703303096450b8b4c8f7a0136f9b4f2357f8c1321f`
freeze passed 137 focused tests, then exposed two stale assertions in the expanded Web
suite. Each fails identically on an isolated archive of committed R3: legacy offsets
are normalized for uniquely grounded literals (accepted since R2B), and a one-member
selection list is valid transport, with an unregistered member correctly returning
404. The tests now verify normalized persisted spans and verified execution, or exact
unavailable-resource rejection with no providers/science. Invalid numeric values,
ambiguous/path inputs and scientific/privacy checks are retained. Production code did
not change to satisfy either test. C/baseline-failure.log and C/baseline-failure2.log
record the baseline reproductions.

That accumulated-process run also exceeded the existing 20-second test polling deadline
in two instrumented rich-API cases, after their scientific steps succeeded and while
presentation was finalizing. The first case passes alone on unchanged source (58.59 s
for its complete test). The diagnostic run was stopped to capture traces: 1,215 passed,
4 failed, 7 skipped, 8 warnings in 1,700.90 s, exit 2; it is not acceptance. All logs and
its unchanged snapshot are retained under C/stale-assertion-and-timing-freeze. The final
regression runs the instrumented rich-API module in a separate process from the other
affected modules. Every previously selected module remains included, with no timeout
increase or test exclusion. The exact process scopes and results are reported above.
C/final-snapshot-delta.json confirms that only the two stale test files differ from the
snapshot used for final native recovery; production and static files are identical.

Further isolated diagnostics retain the timing limitation rather than conceal it:
slower browser-cadence polling still fails (42.47 s), Python-call-only tracing passes
(41.11 s), and an identity-keyed profile counter still reaches the deadline (41.27 s).
These external diagnostic plugins do not modify the repository or final test commands.
The production path, profiler, 20-second deadline and all scientific-call assertions
remain unchanged. C/baseline-rich.log, C/poll-diagnostic.log, C/trace-diagnostic.log and
C/identity-counter.log retain the exact evidence. No further Answerer, presentation
or test-instrumentation work is included in R4.

The result-delivery diagnostic copies the unchanged tests outside Git and changes
only the three expected filename sets to include the already accepted child clustering
output. Both full cases then pass (2 passed, 3 warnings in 27.15 s), including fresh
service reads, exact exports, zero scientific reconstruction, and rejection of the
unretained parent graph. This diagnostic is separate from the frozen regression and
is not added to its counts. C/baseline-delivery.log, C/delivery_expectation_diagnostic.py
and C/delivery-diagnostic.log retain the comparison. No production delivery behavior
or repository delivery assertion is altered to obtain that result.

**Final disposition.** R4's resource changes pass their focused acceptance and real
normal-Web verification. The broad baseline failures above remain explicitly
unresolved; this closeout does not claim an all-green repository regression. Their
R3 reproductions and bounded diagnostics identify no new scientific correctness,
private-data or historical-access regression from the library changes. The final
557-file snapshot is unchanged after every final process. The post-regression
191,897-file protected inventory has zero changes.

The reviewed scope is 32 source/test/static/documentation files. Scientific tool,
model, Registry and authority-schema source is unchanged. `git diff --check` passes;
no dataset, checkpoint, scientific reference, operator configuration, credential,
acceptance fixture or `evals/` file enters the commit. C/final-git.json records the
exact reviewed paths and hashes. The user-authorized closing operation is one
scoped commit, `Complete UA3.6.5-R4 unified scientific resource library`, followed
by a normal push to `origin/main`. C/git-closeout.json records the resolved commit
and the post-push comparison of HEAD, local main, origin/main and queried remote
main; the report avoids a self-referential commit hash. No new work unit begins.
