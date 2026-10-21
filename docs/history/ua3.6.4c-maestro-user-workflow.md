# UA3.6.4c — MAESTRO scientific user workflow integration and acceptance

Assessment: **2026-10-14, Asia/Shanghai**. Overall **PARTIAL**.

A native Firefox resource-question/reply workflow now executes the existing
MAESTRO owner on genuine preserved PBMC data, creates an accepted Run/Revision,
and delivers evidence, a report and a verified annotation H5AD export. The
execution decisions in that successful workflow are **scripted**. A separate
live OpenRouter result question succeeds. The attempted continuous live-model
execution fails before science; these observations are not combined into a live
end-to-end PASS.

Acceptance files are outside Git at
`/home/likeyi/program/ua364c-m8is1ykj` (**R** below). No historical acceptance
Run was copied into a new Session. No upstream scientific computation was repeated.

## A. Git baseline and preservation

HEAD, local `main`, local `origin/main`, and remote `main` agree at
`72aac12dbb748ec3f0c863714a473b321f7681a7`, **Improve scientific parameter
grounding and FASTQ BAM workflows**. Work Unit B is committed. The starting
tracked worktree and index are clean. All 981 Git-listed untracked files belong
to existing `evals/`; the complete eval tree contains 1,274 files.

Remote refs were read using `git ls-remote`, without fetch. An isolated
`git archive HEAD` extraction was used only to reproduce pre-existing test
failures. No reset, clean, fetch, pull, stage, commit or push was performed.

`baseline.json`, `protected-before.json` and `preservation-final.json` record
109,550 initial protected files. The bounded audit hashes files through 2 MiB
and checks size, mode and nanosecond mtime for larger files. Exact full hashes
are additionally checked for scientific resources consumed here. This is not a
fresh byte audit of every historical large dataset/checkpoint.

Protected historical acceptance, Work Units A/B, evals, original datasets,
scientific authority, Sessions/Revisions, external operator configuration,
MAESTRO/EpiZoo sources and previous spending records remain unchanged. A new
Python bytecode cache inadvertently created by importing the historical browser
helper was removed; the original helper and historical file sets are preserved.
The final audit checks 109,540 protected files, including 108,896 SHA-256 checks,
with zero unexpected changes and no additions or removals in the four protected
historical file sets. Exact comparisons are in `preservation-final.json`.

Production changes are confined to:

- `application/resource_continuation.py`, `turns.py`, `turn_decisions.py`, and
  `dialogue_execution.py`: existing shared resource questions now support exact
  structured-input supplementation as well as existing raw registrations.
- `application/annotation_delivery.py`, `matrix_delivery.py`, `interactive.py`,
  `interactive_schemas.py`, and `web/static/app.js`: accepted annotation inventory,
  bounded safe H5AD export, download presentation, and the correct annotation
  contract label in the evidence view.
- `tests/annotation_integration/test_user_workflow.py`: workflow, integrity,
  privacy and bounded-string acceptance.
- Two existing application declaration test files: expectations reconciled with
  already-accepted shared fragment declarations and Work Unit B literal grounding.
- This report.

All changes remain unstaged. Registry remains 23. Scientific annotation sources,
MAESTRO, EpiZoo, compiler, scientific algorithms and authority schemas are unchanged.

## B. Existing MAESTRO readiness

Reviewed M13.2's retained qualification report, M13.3, M13.4, M13.4b, M13.4c and
M13.5 records, scientific owner/contracts, Registry metadata, integration tests,
and the shared InputSet/pending/resource machinery.

`readiness.json` checks the original M11.7 matrix's exact accepted source:

- Run `m117b-pbmc1k-nextgem-r1:run`, step `matrix`, stored revision 14.
- Source authority
  `6999b907cae91e56f1788bb42cf9403bdf99ee9e08b54dd7e7b35fe24ed8167f`.
- Matrix manifest
  `739a8fc2da6344160eb38ba0a3770422e67c0008fe00ddc05abd912e8d70d99e`.
- H5AD `5525e68a1902d5fdd465542e7a2627862c15a95b8f3e3766ca84f2126244dca9`,
  191,009,521 bytes, 1,119 × 1,355,445 canonical int64 CSR.

The existing trusted `AGENT_ANNOTATION_SOURCE_STORE` loader imports its schema-2
scientific authority. A path, uploaded copy, matching row count, or arbitrary
caller-supplied authority JSON cannot establish that eligibility.

The 84,005-byte group TSV has SHA-256
`f9396f8c043bef9874053ead6c1ca5f2c77a90d13322c606bb99c3aededaf595`.
All ordered cell identities match exactly; twelve opaque groups are present.
M13.2 records their independent EpiZoo/neighbors/Leiden derivation before signature
inspection. This task reuses that explicit qualified TSV; it does not infer groups
from an H5AD column or introduce automatic clustering-result correspondence.

The explicit context is **human PBMC**, human/hg38. The original 1,433-byte
`scatac-annotation-inputs.v1` specification is reused with SHA-256
`0ddbca07da8d1cd588b8239e2625b65007df47edcd4748ea1768453e52236ebd`.
The existing owner validates exact run/step, source, grouping, tissue, resource
semantics, namespace and species/assembly bindings.

MAESTRO revision `74f10babf4125b98ef005df4186b14ab02e13bfd` and both executed source
hashes match. The existing runtime reports R 4.6.1, Matrix 1.7.6, dplyr 1.2.1 and
Presto 1.1.0. No packages, scientific sources or production runtime settings change.

## C. Actual user interaction and continuation

The browser selects **Accepted human PBMC hg38 canonical count matrix** and submits:

> Please annotate the cell types in my accepted scATAC-seq dataset using MAESTRO. Use the appropriate available annotation resources and explain any groups that remain unresolved.

The successful scripted-interpreter workflow asks for a reviewed resource choice.
Its user-visible offer is:

> Human PBMC context; preserved independent EpiZoo/Leiden groups; reviewed MAESTRO hg38 genes and human immune signatures; primary annotation only.

The browser clears its input selection and replies:

> Use the reviewed Human PBMC context, preserved independent EpiZoo/Leiden groups, and MAESTRO hg38 genes and human immune signatures. Primary annotation only.

The shared `request_resources` / `answer_resources` mechanism preserves the
original objective, exact source inputs, pending origin/binding/predecessor,
Session generation and captured resource-catalog digest. The resulting Run's
request is the **original request**, not the reply. No reupload or restatement is
needed in this supported path. The exact offered resource handle binds the existing
specification; the researcher never authors internal JSON, hashes or source paths.

Changes to the catalog, source, specification, pending digest or generation fail
closed in tests. An unrelated registered input cannot replace the captured source.
Intervening clarification and fresh-facade reopening preserve the pending/result
contracts. Multiple offers remain distinct; no first-match selection is added.

The menu is an isolated, operator-reviewed deployment of this preserved dataset
and its existing specification. It is not general automated construction of new
annotation specifications for arbitrary source/group/tissue combinations.

## D. Minimal implementation and remaining entry boundaries

Existing InputSets already carry reviewed specification path/digest pairs. The
missing connection was that resource questions only admitted registered raw-data
compositions. The shared extension also offers non-companion InputSets that add
fields without replacing any captured structured value. Only opaque handles and
labels reach the interpreter; selection does not grant scientific authority.
Owner validation remains mandatory at execution.

No annotation resource registry, new Planner, biological inference, per-tool
language parser, workflow completion or pending-task store is introduced. No new
specification builder is needed for this already-reviewed specification.

Annotation output was absent from scientific download inventory. The new delivery
adapter admits exact accepted annotation authority, execution/publication/receipt,
manifest and payload pins. It reuses the existing bounded transfer/snapshot path.
The original annotated H5AD contains a private source-matrix locator; download
therefore produces an explicitly labelled safe derivative, retaining the scientific
arrays, axes, annotations and nonprivate lineage. It does not issue new authority.

Native inspection also exposed the inherited label **Fragment contract** on
annotation evidence. The facade now uses **Annotation contract**, matching the
existing report label. Scientific facts are unchanged.

A separate real 191,009,521-byte H5AD upload succeeds through ordinary HTTP and
binds only `input_path` (`ordinary-entry.json`). Even a byte-identical copy of the
historical canonical source acquires no source Run authority by registration.
Existing `inspect_scATAC` can inspect such an input. Constrained external adoption
can establish its own explicitly external matrix contract, but **M13.5 rejects
external adoption**, including externally declared fragment counts. It cannot
certify the historical canonical fragment-record counting method by assertion.
Arbitrary peak/binary/insertion matrices therefore have no supported direct
MAESTRO entry under the present owner. The actionable scientific alternatives are
selection of a qualified canonical result plus exact groups/resources, or a
supported upstream fragment-to-canonical-matrix workflow with those prerequisites.
No values are converted or relabelled to manufacture compatibility.

Group TSV is not an admitted upload type: an attempted `input_type=groups` returns
HTTP 422. General new-file supplementation for grouping is **not implemented or
qualified**. An operator can provision a separately reviewed exact group/specification
choice; no historical choice is silently replaced. Enabling arbitrary group
uploads would require a reviewed file-role boundary and specification binding,
which this bounded existing-resource integration does not invent.

## E. Genuine scientific execution

The native-browser successful execution takes **322.84 seconds**, including
presentation, using scripted language/planning decisions and the unchanged real
registered scientific owner:

- Session `f2751a85-cd30-49a5-ad2b-8e204d64fb0f`.
- Run `turn-d97f68e3deccb3894f9c47ece3b35c95aea5d2d493bd2813a57c3242bc49f2f4:run`.
- Step `annotation`, tool `annotate_scATAC_cell_types`.
- Revision `35df30cc96723af0c05621a12278d756f66d75f5e34b8c2f0a87448fd5908918`,
  generation 1.
- Annotation authority
  `74d5a7865c234658a5393887acda16e493ba203bb366ec63683724cd2745c499`.
- Annotation manifest
  `1b141d8bfe7b3ce0258dbea349fcbe9b545ce271dc82a346828b79c7e51c12c0`.

Exact resolved arguments and every output sidecar size/hash are recorded in
`scientific-acceptance.json`. Normal Interpreter/Planner, compiler, preflight,
publication, independent owner verification, evidence/report and Session completion
are used. Instrumentation records **two primary backend calls**, one for production
and one for **one annotation-owner verification**. This is the established pinned
backend replay verifier, not an independently implemented annotation algorithm.
Covered upstream verifiers and EpiZoo remain at zero; no upstream producer is in
the executed plan. CellMarker and EpiAgent are not invoked.

Actual new output: **1,119 cells; 12 groups; 1,090 assigned; 29 unassigned;
0 ambiguous**. Every per-cell primary record also matches the preserved M13.4b
record, as a post-execution comparison, not a hardcoded execution target.
Source CSR data/indices/indptr, complete cell/feature identities and order, and
original obs/var metadata are exactly preserved. Source H5AD SHA-256 remains exact.

`audit_result.py` performs a fresh-process accepted-result audit and terminal
Application resume with scientific-entry/provider guards. Resume preserves all
29 then-existing workspace files and executes zero science.

## F. Biological scope and unresolved cells

Labels remain method-derived primary annotations, not experimentally validated
cell identities. The result retains `validation_state=not_assessed`.

| Group | Cells | Primary annotation | Status |
| --- | ---: | --- | --- |
| 0 | 60 | NaiveBcells | assigned |
| 1 | 160 | Monocytes | assigned |
| 4 | 77 | ActNK | assigned |
| 2 | 114 | Monocytes | assigned |
| 6 | 108 | RestMemCD4Tcells | assigned |
| 7 | 107 | ActNK | assigned |
| 8 | 122 | RestMemCD4Tcells | assigned |
| 9 | 175 | NaiveCD4Tcells | assigned |
| 5 | 106 | MacrophagesM0 | assigned |
| 10 | 29 | null | unresolved |
| 11 | 37 | MemoryBcells | assigned |
| 3 | 24 | Monocytes | assigned |

Rows retain owner order. Group `10` has reason `nonpositive_maximum`, top score
0.0, and null candidate/primary annotation. Its zero-score alternatives do not
constitute an exact **positive** tie, so the status is unresolved, not ambiguous.
The native detailed-evidence view exposes this reason, competing scores and
signature coverage with explicit bounded omissions. No threshold is retuned and
no group number is interpreted as a cell type.

Earlier CellMarker/EpiAgent observations remain historical, separate optional
corroboration. They neither validate every current label nor rescue nulls.
Sparse evidence, T/NK competition, subtype accuracy, purity and held-out accuracy
retain the limitations documented in M13.2/M13.4b/M13.4c. No new confidence,
ontology mapping, biological truth or cross-species claim is introduced.

## G. Results and delivery

The genuine newly generated annotated H5AD is **348,175,550 bytes**, SHA-256
`ba172ed206186498c956cf1d746538d58169ea57473bd031bca7db8427faa547`.
The independent byte identity agrees with historical M13.5, although this is a
new execution with its own publication and authority.

The delivered safe export is **348,087,949 bytes**, SHA-256
`009205d4085a82b2fa350857c3aaa320e3a099c3d7a64cad5f073673d9f64e14`,
retained at `R/annotated-export.h5ad`. It omits only the private
`cell_type_annotation.matrix_manifest_path` from source metadata and adds a closed
export provenance record. HTTP headers distinguish source/content hashes.
This is not advertised as the original H5AD's unchanged bytes.

Native Firefox fetches and hashes the actual export (HTTP 200, about 30.58 seconds
including adjacent browser checks). A separate bounded HTTP download takes 31.18
seconds including full sparse/axis comparison. Both delivered hashes agree.
`download-check.json` verifies exact CSR arrays, complete obs/var, categorical
annotations/nulls and remaining annotation lineage against the accepted original.

Reports and evidence are pinned accepted-result projections. The rendered safe
report contains 3,792 characters; raw server paths remain private. Native controls
show the group summary, assigned/unassigned counts, validation state, unresolved
rationale and scientific download links in both conversation/result views.
The original report, accepted evidence and all primary sidecars remain in the new
managed scientific workspace; paths and hashes are in `scientific-acceptance.json`.

The first delivery implementation exposed excessive fixed-width string copying
on the full cCRE axis, causing 30-second socket and 120-second browser-driver
timeouts. The exporter now probes bounded short strings and rereads saturated
probes individually at the existing 65,536-byte limit. Exact Unicode/long-value
and excessive-length tests pass. No scientific payload or verification was removed.
The terminal-only server holding obsolete download workers was stopped before
restarting the corrected delivery; no scientific stage was interrupted or replayed.

## H. Provider qualification and spending

OpenRouter `openai/gpt-5.6-sol` is the isolated development-primary model. Groq
remains independently selectable in configuration but is not called. Credential
availability is checked as booleans; no key is printed or persisted. All prior
ledgers and the production operator JSON remain unchanged.

The new ledger enforces 12 total completions, six per turn, $3.00 conservative
pre-dispatch reservations, 4,096 output tokens, zero SDK retries, no silent fallback,
and no refunds of reservations. Each dispatch checks the remaining budget and
persists its reservation before contacting the provider. Eleven guard tests pass.

The first live Session is `7d0d63e9-86b0-4dbd-8c75-329bc6dbc761`. Its ordinary
browser request is exactly the request in section C. Actual decisions:

1. Interpreter chooses `execute_plan / annotate_scATAC_cell_types`, despite an
   offered resource-choice route.
2. Capability selection chooses `marker_annotation`.
3. Planner omits the required specification port; compilation rejects the plan.
4. The existing bounded planner repair call returns `unsupported`, explicitly
   stating that the required specification is unavailable to that plan.

The browser displays `UNSUPPORTED_REQUEST`; no science or Revision is created.
This exposes a remaining user-facing gap: the available resource choice was not
connected through a live clarification, and generic failure text is less useful
than the existing resource-question path. It is **not** proof that this qualified
PBMC source is scientifically unsupported. No manual replacement, semantic override,
additional favorable-attempt retry, or automatic resource injection follows.
The ordinary compiler-repair completion is counted explicitly; it is not an SDK retry.

After the separately scripted-admitted real result, native Firefox selects the
real OpenRouter profile and asks:

> How many cells received a primary annotation, how many remain unassigned, and were these cell identities independently validated?

Two live calls (Interpreter and scientific Answerer) produce an accepted answer
with exact accepted-evidence claims for 1,090 assigned, 29 unassigned, and
`not_assessed`. It correctly declines to establish independent biological
validation. Its broader insufficient-evidence conclusion remains visible.
This is successful **separate live result dialogue**, not continuation of a
successful live scientific-execution chain.

Final accounting: **6 completions**, **$0.459926 reserved**,
**$0.0518320 provider-reported cost**, zero unfinished reservations and no halted
ledger. Per-request prompts, schemas, raw decisions, routing/usage observations
and failures are retained in `R/provider`; `budget-summary.json` consolidates them.
The inspected provider/scripted prompt surfaces expose no consumed resource paths.

## I. Tests, native Firefox and recovery

No full repository suite is run. Initial affected regression: **431 passed,
5 failed, 3 warnings in 343.11 seconds**, over 436 tests. A frozen comparison
checks 496 source/test/static files unchanged through the initial verification
point. Every one of the five failures reproduces on an untouched isolated archive
of HEAD (5 failed in 3.16 seconds): old assertions excluded already-accepted
fragment choices, shared parameter answers, namespace strings or unique-literal
offset grounding. Production behavior is not weakened to satisfy old expectations.

The two stale test files are reconciled with accepted current contracts, retaining
invalid choice, substituted role/value and missing-literal rejection tests.
Their rerun yields **86 passed, one remaining stale schema assertion**; the final
corrected evidence-quote schema assertion passes separately. This is reported as
split validation, not a falsely claimed single clean 436-test invocation.

After delivery optimization, the final affected delivery/workflow/privacy/static
batch passes **90 tests in 77.80 seconds**, with three existing warnings. After the
annotation label correction, workflow/presentation checks pass **7 tests in
9.19 seconds**. The initial 14 existing resource-continuation tests and 10 initial
new workflow tests also passed during development. Test invocations, logs and
frozen inventories remain in R. Initial harness issues (base-environment pytest,
HTTP field naming and premature presentation polling) are retained as development
failures, not scientific acceptance results.

Coverage includes exact resource choice/captured identity, stale answers/catalog,
source/specification substitution, scientific owner authority and group/resource
validation, output preservation, export tampering/private fields/links/codes,
Unicode and long identifiers, shared declaration/role continuation, original and
processed H5AD delivery, EpiZoo declarations, and FASTQ/BAM resource handoff.
Synthetic scientific tests use their explicitly stubbed native annotation backend;
the genuine PBMC execution above uses actual MAESTRO/R production and replay.

Native Firefox (headless browser engine controlled by Marionette) exercises the
rendered input menu, composer, resource question/reply, long-running status,
result history, evidence/detail/report controls and refresh. Actual delivered
bytes are fetched in Firefox; binary equality receives a separate HTTP/file audit.
This is genuine browser-engine interaction with automation, not a claim of manual
human operation or merely an HTTP observer.

Observed statuses are request received, analysis in progress, finalizing and
analysis completed. No numeric progress percentage is invented.
`native-reopen.json` records ten read-only lifecycle checks in a fresh server:
six exact conversation messages, same Revision/generation, history navigation,
annotation rationale, download links, refresh, zero provider construction, zero
scientific entries, and **30 unchanged workspace files**. No new biological
execution, branch or crash-during-science recovery is claimed for these checks.
The final native evidence view uses the corrected annotation contract caption.

## J. Resources, acquisition and compute

No new public dataset, reference, checkpoint or package is downloaded. Existing
qualified resources are consumed at their preserved locations:

| Resource | Bytes | SHA-256 | Qualification |
| --- | ---: | --- | --- |
| Canonical PBMC source H5AD | 191,009,521 | `5525e68a1902d5fdd465542e7a2627862c15a95b8f3e3766ca84f2126244dca9` | Exact M11 accepted canonical human/hg38 matrix and authority |
| Independent group TSV | 84,005 | `f9396f8c043bef9874053ead6c1ca5f2c77a90d13322c606bb99c3aededaf595` | M13.2 independent groups; exact current cell order |
| MAESTRO GRCh38 refgenes | 25,673,202 | `9b18e56531155226450254aefd192bb7f49d03d74c5bbfdc53af5411a0d2fb37` | Pinned MAESTRO transcript/exon resource; human genomic annotation |
| Human immune signature TSV | 21,266 | `3c202410d4f46049a298784838e5f9d214fc67fcddf34536d1c3811e383b601a` | M13.5 header-only export of retained MAESTRO human.immune.CIBERSORT; human PBMC |
| Annotation input specification | 1,433 | `0ddbca07da8d1cd588b8239e2625b65007df47edcd4748ea1768453e52236ebd` | Exact M13.5 run/step, groups, context and resources |

The publisher/source revision is the retained MAESTRO release above. Historical
acquisition/source records stay in M13.2; this task's reuse checks are dated
2026-10-14. No independent RefSeq release designation or new biological
qualification is invented. Source Python/R and executable hashes/sizes are also
recorded in `readiness.json`; historical library dependencies remain in place.

Before execution: approximately 15 TiB free on the scientific workspace filesystem,
45 GiB on `/tmp`, and 115 GiB available host memory; expected new science was about
420 seconds and 2 GiB scratch/output. Actual browser scientific execution was
322.84 seconds. An instrumented main-process high-water observation was
3,599,408 KiB; this is not a complete child-process peak measurement. No GPU,
full-matrix densification or >20-minute scientific stage is used. The complete
new evidence/test workspace is approximately 1.13 GiB at closeout.
New acquisition cost is zero; new provider cost is recorded in section H.

Deployment remains explicit: review `operator-scientific.json` / `operator-isolated.json`
and `readiness.py`'s existing annotation runtime/source-store bindings before
configuring a production menu. The task does not install a production default.
The reviewed PBMC specification is eligible only for its exact source/group/context;
other sources require their own qualified bindings.

## K. Final qualification

| Scope | Assessment | Meaning |
| --- | --- | --- |
| Technical software workflow | **PASS, bounded** | Shared structured resource question/reply, exact retained task/source, real owner publication, evidence and downloads; scripted interpretation |
| Genuine scientific data execution | **PASS, bounded** | New actual PBMC MAESTRO production plus independent replay; exact source/derivative conservation |
| Real LLM semantic observations | **PARTIAL** | Initial execution entry fails; separate evidence-grounded result answer succeeds |
| Complete continuous live-model workflow | **BLOCKED in the demonstrated attempt** | No live resource clarification-to-execution chain; no substitute success claimed |
| Arbitrary uploaded H5AD / newly uploaded groups | **BLOCKED under current entry/owner scope** | Registration/adoption cannot supply canonical counting authority; group TSV upload unsupported |
| Overall UA3.6.4c | **PARTIAL** | Scientific/software integration is demonstrated; full live and general file-supplementation usability remain unqualified |

## L. Carryover and next recommendation

Preserve UA3.6.4a's unresolved live cross-Revision dialogue compliance/expressiveness
qualification in UA3.6.5. Preserve UA3.6.4b's genuine FASTQ/BAM production and complete
live raw-to-result qualification gaps, and the lack of durable objective continuation
before a FASTQ library has been completed. Genuine Work Unit A fragments results
and all previous provider outcomes remain intact.

MAESTRO-specific carryover is the observed live failure to choose the available
resource prerequisite, clearer treatment of missing required specifications when
planning terminates unsupported, and a reviewed shared group-file entry/binding
path for researchers who supply new grouping files. General automatic specification
assembly for new source/group/tissue combinations is not claimed. These gaps should
not be solved by overriding provider semantics, inventing authority, requiring
optional validators, or broadening arbitrary H5AD count semantics.

The independent UA3.6.4d cross-species scenario may proceed after review, subject
to its own qualified resources and public-tool scope. This human/hg38 result neither
qualifies cross-species MAESTRO nor blocks unrelated supported neutral workflows.
Neither UA3.6.4d nor UA3.6.5 is started here. Stop at this report; no commit or push.
