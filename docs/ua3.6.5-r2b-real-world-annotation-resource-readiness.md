# UA3.6.5-R2B — Real-world annotation and resource readiness

2026-10-19 (Asia/Shanghai). **R2B PARTIAL.** Four bounded product repairs, a test-contract cleanup and a live native
positive-control chain are demonstrated. The full external matrix does not
complete annotation; conversational evidence answers and production readiness
remain incomplete. Evidence root **R**:
`/home/likeyi/agent-ua365-r2b-6zkcjv2c`, outside Git. UA3.6.5 remains open.

## Baseline and reviewed boundaries

Initial branch `main`; HEAD, local `main`, `origin/main` and the read-only remote
`main` query all return `d5320bebb6aaa7e812cc3f3f173b4fa04b5cf862`.
R1 `d46e132b27fa0ad80c5859bedcf67fd6639dd47d` is its parent. The tracked tree
and index were clean; `?? evals/` was the sole initial status entry.
`R/baseline.json` records 720 tracked-file hashes and 123,109 protected entries.
Protection uses size/mtime/mode plus hashes for files ≤64 KiB and every eval file;
it is not a new full-byte audit of all historical biological datasets/models.

Reviewed README/AGENTS and the USI-4 handoff/resource closeout, UA3.6.4c native
peak acceptance and peak-contract review, R1 and R2A records; inspected current
annotation input, resource binding/discovery, scientific owner, Registry,
capability selection, Session continuation and error-policy code. Historical
scientific qualification is distinguished from new execution throughout.

The existing direct method remains `maestro-enhanced-count-weighted-peaks.v1`:
human hg38 raw Cell Ranger ATAC cut-site counts, qualified primary nuclear peaks,
explicit exact groups, compatible refgenes/signatures and tissue context. Native
Enhanced RP, marker statistics, scoring, conservative assignment, independent
owner replay and publication are reused. No new tool, grouping engine, reference
catalog, scientific method or UI layout is introduced.

## Authentic external input

New acquisition is the official [10x 1k PBMC Next GEM dataset](https://www.10xgenomics.com/datasets/1-k-peripheral-blood-mononuclear-cells-pbm-cs-from-a-healthy-donor-next-gem-v-1-1-1-1-standard-2-0-0),
published 2021-05-03, Cell Ranger ATAC 2.0.0, human PBMC, CC BY 4.0. The publisher
reports 1,004 detected nuclei. This is a newly acquired publisher dataset,
separate from the 484-cell acceptance input; the publisher states the PBMC loads
share nuclei preparation, so no independent-donor replication is claimed.

All files come from
`https://cf.10xgenomics.com/samples/cell-atac/2.0.0/atac_pbmc_1k_nextgem/`:

| Suffix after `atac_pbmc_1k_nextgem` | Bytes | SHA-256 |
| --- | ---: | --- |
| `_filtered_peak_bc_matrix.h5` | 12,853,716 | `a1160985dd9dbec14e5c4ccb2f6baabaf12be4e2d7d914b6d4ad7f3543bd1992` |
| `_analysis.tar.gz` | 63,996,499 | `392d0796df58cb95e225059fea1307715d64ef5813334e159b98e0e7986d13b4` |
| `_web_summary.html` | 2,674,277 | `48de685573766fa677034db6c9a23fa18e8ba3b2ad1118b7474d4fbc0fd10850` |

`R/acquire.py` checks free disk (about 15.6 TB), streams with connection/read and
elapsed bounds, checks Content-Length, caps each file at 250 MiB and hashes exact
bytes. `R/acquisition.json` records URLs/dates/roles; no publisher checksum was
available in the inspected evidence. Total new scientific download: 79,524,492
bytes. No reference/model download was necessary.

Publisher feature metadata explicitly says GRCh38. The
[10x matrix contract](https://www.10xgenomics.com/support/software/cell-ranger-atac/latest/analysis/feature-barcode-matrices)
establishes features × barcodes and cut-site counts; peak IDs retain publisher
zero-based half-open coordinates and barcode suffixes. `R/prepare_data.py`
transposes sparse storage into ordinary cells × peaks AnnData and imports
`analysis/clustering/graphclust/clusters.csv` only after exact ordered barcode
comparison. It retains every cell, peak and count, without filtering, alias
repair, normalization, annotation-specific metadata or invented provenance.
This is documented standard-format preparation, not an original publisher H5AD.

Prepared file: 1,004 × 82,579 int32 CSR, 8,164,121 nonzeros, 22,279,127 total
counts, maximum 264; six publisher groups (359/178/132/115/114/106 cells).
Size 26,869,587; SHA-256
`c75bca64515710f6045a1c65d9f0dfb9fe9827102501856007b2bfcb58b33c81`.
`R/preparation.json` records round-trip sparse equality and exact axes/groups.
`R/source-scope.json` identifies 31 peaks (6,062 counts) outside the existing
qualified primary scope. They are not silently classified by name or removed.
The full publisher matrix therefore does **not** complete the current direct
annotation method. No acceptance-tailored primary subset is manufactured.

## Resource readiness and deployment

| Resource | Physical / qualification | Normal Web visibility |
| --- | --- | --- |
| MAESTRO GRCh38 refgenes, revision `74f10babf4125b98ef005df4186b14ab02e13bfd` | Existing pinned source; SHA `9b18e56531155226450254aefd192bb7f49d03d74c5bbfdc53af5411a0d2fb37` | Not configured |
| Human immune CIBERSORT signature export, same revision | Existing documented header-only export; SHA `3c202410d4f46049a298784838e5f9d214fc67fcddf34536d1c3811e383b601a` | Not configured |
| Qualified hg38 GENCODE-49 chromosome/QC bundle | Existing owner catalog; manifest SHA `e9037c7a80964495e5b3aaea4aa6d634035d05cf7aaf5a36d8cff27bb21afbd5` | Not offered as an annotation dependency |
| Native MAESTRO / R / Presto runtime | Existing historical installation and pinned sources | Required annotation environment absent from the inspected normal conda launch |

The production JSON has two InputSets (accepted PBMC cCRE input and mouse species)
and EpiZoo resources. Its three local-resource records are private H5AD uploads,
not qualified annotation references. Its default conversational profile is Groq;
Firefox explicitly selects the configured OpenRouter GPT-5.6 Sol profile.
Physical presence, scientific qualification and deployment availability differ.

`R/resource_readiness.py` freshly validates exact existing gene/signature/scope
pins using the existing owner: 28,307 symbols, 78,362 primary transcripts and 28
candidates. This validates computational compatibility and declared applicability,
not biological truth or newly reconstructed genome qualification. Pinned native
source checks pass. A first diagnostic incorrectly assumed a `Runtime.check()`
method and failed; the corrected diagnostic uses the existing checks.

`R/operator-initial.json` preserves the production-equivalent isolated config.
`R/operator.json` adds the existing reviewed human PBMC reference through the
existing InputSet mechanism; `R/scientific-environment.json` supplies the existing
catalog/runtime only to the isolated server. No production file/default changes,
private-resource promotions or cross-user redistribution occur. The pending
external task discovers this added resource after restart without reupload.
Production deployment requires explicitly reviewing/enabling these existing
resource and runtime pointers; this acceptance configuration is not deployment.

## Real browser/model observations

All described live turns use native headless Firefox/Marionette, ordinary upload,
composer and HTTP application paths, and the configured
`openai/gpt-5.6-sol:floor` transport/routing/retry/price limits. Credentials are read
from the existing environment and never copied into evidence. Provider
prompt/schema/response audit and science-entry counters are observational;
no response or scientific implementation is replaced. Audit calls are not billed
cost measurements.

| Trace | Request / observed result |
| --- | --- |
| A1 | “Can you annotate the cell types in this dataset? These are human PBMCs.” Correct typed annotation request and targeted assembly/history/count-slot/grouping clarification. |
| A2 | Natural reply identifies 10x/Cell Ranger 2.0.0, hg38, original cut-site counts in X and publisher `cellranger_graphclust`. The model emits prose measurement history; Agent incorrectly reports incompatible declarations because it expects an undisclosed exact identifier. No science. |
| A3 | After the handoff repair and isolated resource activation, a natural correction continues the **same** task/source. The model selects the canonical measurement and exact offered `human PBMC` context/reference. Planning reaches the owner; full external peaks fail scope qualification before any native annotation. Initial UI gives only “The operation failed.” |
| B1–B2 | Historical genuine 484-cell positive control, ordinary annotation request and grouping reply. Interpreter correctly binds annotation. Capability selection incorrectly offers label transfer only; detailed planning returns unsupported. No native science. |
| C1–C2 | Unchanged original control request and grouping reply after the capability-description correction. The selector includes marker annotation, Planner binds the original counts/groups, and native production plus one independent owner verification succeed. |
| C3–C4 | “Which groups remain unresolved, and what evidence supports these annotations?” and “How many cells were assigned, and were any groups unresolved?” Both correctly select the accepted result but fail answer-generation factual binding. Zero additional science. |
| A4 | Reopen the original external Session and request annotation again with the same explicit scientific context. Correct canonical measurement/reference binding; Firefox now reports `ANNOTATION_PEAK_SCOPE_UNSUPPORTED` with the actual primary-scope limitation. No native backend, annotation result or Revision. |

Exact utterances, decisions, screenshots, timelines, Session/Run identities and
errors are retained in `R/*-page.txt`, named browser JSON files,
`provider-audit.jsonl`, `science-audit.jsonl`, and `workspace/`.
The historical control is explicitly separate from external-user acceptance.
Across ten live turns there are **23 actual provider calls**: ten interpreter,
four capability selection, four detailed planning, three output selection and
two answerer calls. All use the configured model; no transport errors or model
fallbacks occur (`R/audit-summary.json`). These bounded trials do not establish
provider reliability or a causal statistical estimate for the metadata change.

## Findings and owning repairs

| Severity | Finding / disposition |
| --- | --- |
| P1, repaired | `measurement` accepted free text while deterministic admission required an identifier absent from prompt/schema. Schema now exposes the owner constant and its supported meaning, retains unknown/null and unsupported text, and leaves language interpretation to the LLM. No regex repair or relabelling of binary/normalized counts. |
| P1, repaired | Pending annotation declarations could not correct a wrong measurement/context. Typed corrections now create a new record for species/assembly/measurement/context; captured source/group file handles remain fixed. Source-metadata merge stays strict. Original records and accepted evidence are not rewritten. |
| P1, repaired metadata | Capability selector still described marker annotation as canonical-only. Updated its existing description to reflect direct peak H5AD, explicit groups, native MAESTRO and gene/signature resources. No forced selection, auto-completed workflow or second planner. Live quality remains separately measured. |
| P2, repaired | Scope rejection was hidden by generic `ValueError` classification. The coordinate owner now raises a typed scope error for nonmember contigs; the existing classifier/error catalog supplies a safe actionable message. Coordinate bounds, scope membership and algorithm remain unchanged. |
| P2, unresolved | Both live evidence answers repeat the categorical status `assigned` in prose. All claim IDs are valid, but the existing whole-claim contract rejects that prose. Offline replay identifies `factual_binding/unbound_categorical_value` in both; no evidence corruption or wrong current-result selection. The report/detail/download routes remain usable. |
| P2, recorded | The visible summary prioritizes an artifact-schema number and omitted-record count, while labels require expanded details; failed read-only answers display “Analysis failed” beside a still-successful result. No UI redesign or patch is made. |
| Deployment gap | Qualified resources/runtime exist but are not available to ordinary production Sessions. Isolated activation succeeds at binding; production activation remains explicit. |
| Scientific/product limit | Full external publisher peak axis is outside current primary scope. This is an honest method-boundary failure. Supporting an explicit provenance-preserving preparation or broader qualified scope requires a separate reviewed capability decision; no silent filtering. |

No P0 scientific-integrity or privacy failure was observed. The repeated answer
failure was investigated at the shared renderer, not dismissed as an isolated
response. `R/answer_diagnostics.py` replays captured responses without providers
or science. The categorical guard treats even ordinary predicate wording as a
value repetition; the provider nevertheless violates its explicit current
contract. Permitting prose reuse safely requires reviewed subject/value
correspondence rules, not a special exemption for “assigned.” This unresolved
shared product limitation is part of the PARTIAL disposition.

Changed owners: `application/raw_handoff.py`, `application/turn_decisions.py`,
`orchestration/planning_scope.py`, `orchestration/annotation_registry.py`,
`orchestration/error_policy.py`, and the typed exception in
`tools/analysis/external_marker_annotation.py`. No native calculations, EpiZoo,
Registry membership, data or UI source changes. Focused tests are in the existing
annotation handoff/external-entry files. The initial-scientific-declarations
tests also replace obsolete fixed parameter totals with exact offered-pair
coverage and uniqueness checks. No new production module or compatibility
writer is added.

## Accepted control, downloads and recovery

The control is the previously prepared genuine 484-cell/65,887-primary-peak
H5AD (source SHA `72e581c3f027cc9330098f655335b6cc9406d39c136c882d4e2e95768c2efe38`),
with all 65,908 original publisher peaks preserved in raw. Its historical primary
selection is disclosed in UA3.6.4c; R2B does not recreate or tailor this source.
The new value is the live request/clarification/planning/native-result chain.

- Session: `ceccb6f2-7fee-46f9-8673-50fd0551a690`.
- Run: `turn-efaccb4900ddc3676237e76ec1c2bd50a7461a63c5191ef481233e3852d774e0:run`.
- Revision: `277a861fb29a3410fba4c3c405f5a4aa1443e9488dd78e5ac76c6003df56642f`.
- Manifest SHA: `7d5dc351f091ab96504aaae89e115f73abbf3b45b986e99b62b07bf3bd69a442`.

Exactly **two native backend calls**, one production and one within **one owner
verifier entry**, are observed. Four source-qualification entries cover the two
external failures plus control production and verification. No upstream
preprocessing, PCA, EpiZoo, implicit clustering or optional annotation validation
runs. The method labels all 484 cells: group 1 Monocytes (207), group 2 ActNK
(114), group 4 MemoryBcells (57), group 3 NaiveCD4Tcells (106); zero unassigned or
ambiguous. These match the historical control group summaries and remain
`validation_state=not_assessed`, not biological ground truth.

Firefox displays the completion summary, report, detail evidence and native
H5AD download. No scientific figure is produced by this annotation path.
The 22,903-character delivered report is explicitly a safe technical report.
The native download is 91,020,412 bytes, SHA-256
`6c694d33aacab55dd87276d32014e7e3e126dc8723c97ddc6af5fa7a1fea7f24`,
matching the browser response and `X-Artifact-Content-SHA256`. It intentionally
differs from source publication SHA
`accb3bcffc42e50943806898a86f0bdd32759039b2f8363c83b722ba545187c9`
because the established export omits the private source locator.
`R/science_check.py` reopens the actual download and checks exact axes, original
obs/var, sparse X/raw and every group-to-cell label/status. No backend rerun.

Refresh and a fresh Firefox/server restore all eight conversation texts,
generation 1, the same Revision and report. **All 71 workspace files remain
byte-identical, with zero provider/scientific calls** (`R/recovery.json`).
Earlier failed Sessions remain accessible; the original USI-4 accepted Session
also passes the separate read-only check below. No branch/comparison acceptance
or clustering-to-annotation grouping handoff is claimed in this round.

## Validation and closeout

The initial handoff run had 17 passes and one new-test failure because the test
assumed unavailable `jsonschema`. Removed that dependency; final handoff run:
**18 passed in 71.90 seconds**, exit 0. It covers pending correction after restart,
original record preservation, exact source identity, source substitution rejection,
strict metadata conflicts and unsupported measurements. These are scripted,
small/simulated software tests, not live biological qualification.

`R/historical_read.py` reopens the preserved USI-4 Session, Revision, artifact
inventory and accepted evidence: HTTP 200, zero provider/scientific calls and all
38 workspace files byte-identical. Source exception typing does not rewrite or
recompute historical results. New consumption of private qualified resources
continues to enforce their existing owner-version pins.

The first adjacent regression was intentionally interrupted to incorporate the
newly observed capability-description defect; it is not an acceptance pass.
Its log/result/source freeze remain under `R/regression-interrupted/`.
The final production snapshot's adjacent suite records **395 passed, three
failed, one warning in 704.07 seconds**, exit 1; no skips or test exclusions,
all `RUN_*` gates disabled, all 550 frozen source/test/static files unchanged.
It covers all annotation integration, capability-scoped planning, scientific
parameter admission, pending roles, FASTQ resources, qualified-resource discovery,
Web resource continuation and Session upload reuse. Exact argv, log and XML:
`R/regression-command.json`, `R/regression.log`, `R/regression.xml`.

All three failures reproduce on an untouched `git archive` of `d5320be` in
**1.69 seconds**. They are obsolete parameter totals in
`tests/application/test_initial_scientific_declarations.py`:
`test_reviewed_species_and_fragment_choices_are_offered_in_closed_schema`, and
`test_pending_scalar_schema_preserves_exact_pairs_and_scoped_choices` for
`species-answer_prerequisite` and `parameters-answer_parameters`. The actual
parameter sets are correct; totals are 29/26 rather than the historical 24/21,
and the numeric-only fixed total is also stale. Their baseline reproduction is
preserved in `R/baseline-failures*`.

Removed those fixed-size assumptions and retained/strengthened exact
projection coverage, uniqueness, offered finite choices and closed schemas.
The complete affected module then passes **56 tests in 26.56 seconds**, exit 0
(`R/parameter-final.log`, XML and command record). Only that test file changes
after the frozen adjacent run; production code is identical. All 398 selected
cases therefore have passing coverage across the adjacent and corrected-module
runs; there is **no claim of one final all-green 398-test invocation**, and counts
overlap. The warning is the intentional duplicate-cell-name negative fixture.
No full historical scientific suite or large biological recomputation is run.

Two harness limitations are retained in the evidence: the first control browser
launch raced server startup and failed before submission/provider dispatch; its
normal retry is logged separately. The initial resource-readiness diagnostic
failure and unavailable test-only `jsonschema` import are described above. Neither
is counted as a successful product trial.

### Preservation, Git state and remaining work

`R/preservation.json` finds **zero changes among 123,109 protected entries** under
the stated bounded policy, preserves all **1,274 eval files** and their inventory,
and checks exact hashes of acquired originals and the converted external H5AD.
The original production configuration, scientific references, historical Sessions,
R1/R2A evidence and checkpoints are not replaced. Native EpiZoo is unchanged.
Both isolated server and Firefox are stopped after acceptance.

Final branch is `main`; HEAD, local `main`, `origin/main` and freshly queried
remote `main` remain `d5320bebb6aaa7e812cc3f3f173b4fa04b5cf862`.
`git diff --check` passes. The index is empty. Nothing is staged, committed or
pushed. `R/final-git.json` records exact final hashes/status:

```text
 M src/agent/application/raw_handoff.py
 M src/agent/application/turn_decisions.py
 M src/agent/orchestration/annotation_registry.py
 M src/agent/orchestration/error_policy.py
 M src/agent/orchestration/planning_scope.py
 M src/agent/tools/analysis/external_marker_annotation.py
 M tests/annotation_integration/test_conversational_handoff.py
 M tests/annotation_integration/test_external_entry.py
 M tests/application/test_initial_scientific_declarations.py
?? docs/ua3.6.5-r2b-real-world-annotation-resource-readiness.md
?? evals/
```

**R2B is PARTIAL; UA3.6.5 remains ongoing.** The original external input cannot
complete the qualified primary-scope method, the normal production deployment
still lacks activated annotation resources/runtime, and both live scientific
answers fail their current generation contract. Control execution, accepted
results/downloads and recovery must not be presented as resolving those limits.
No universal H5AD/MAESTRO compatibility or independent biological accuracy is
claimed.

The highest-value next exploration is the shared scientific-answer boundary:
make ordinary supported explanations usable while preserving exact
subject/value attribution, using the captured two-response reproducer before
changing its factual checks. Separately review an explicit, provenance-preserving
primary-scope preparation capability for full publisher matrices and the concrete
resource/runtime deployment above. Accepted-clustering-to-original-count annotation
remains a valuable subsequent journey; this round does not establish or refute
that handoff.

## Authorized Part A closeout — 2026-10-19

The subsequent user request authorizes two separate commits and normal pushes.
Before Part A, HEAD/main/origin/main/remote main all matched `d5320be`; the exact
nine modified source/test files and this report matched `R/final-git.json`.
Review confirms the four repairs preserve scientific ownership and captured
source identities. The recorded regression remains applicable with its separate,
overlapping corrected-module run; it is not one all-green regression invocation.

New evidence root **F**: `/home/likeyi/agent-ua365-r2b-closeout-5np37837`.
The original R evidence is preserved. Final focused handoff and scope-error
checks pass **19 tests in 74.92 seconds**, exit 0 (`F/part-a.log`, `F/part-a.xml`).
`git diff --check` passes. Only the ten reviewed R2B source/test/report files are
included in the authorized Part A commit, titled
`Complete UA3.6.5-R2B annotation handoff corrections`. Evals and scientific data
are excluded. Product acceptance remains **PARTIAL**. The earlier unstaged Git
record above describes the original exploration checkpoint.
